
import os
import csv
import time
import threading
from datetime import datetime
from collections import deque

import av
import cv2
import mediapipe as mp
import numpy as np
from streamlit_webrtc import VideoProcessorBase


# ---------------- SETTINGS ----------------

CALIBRATION_SAMPLES = 20
BLINK_THRESHOLD = 0.21
MINIMUM_CLOSED_FRAMES = 2
SMOOTHING_FRAMES = 5

GAZE_HORIZONTAL_DELTA = 0.07
GAZE_RIGHT_DELTA = 0.03
GAZE_VERTICAL_UP_DELTA = -0.025
GAZE_VERTICAL_DOWN_DELTA = 0.025

LOOKAWAY_WARNING_SECONDS = 1.0
LONG_EYE_CLOSURE_SECONDS = 2.0
YAW_THRESHOLD = 15
PITCH_THRESHOLD = 12
HIDE_FACE = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "session_logs")
SUMMARY_FILE = os.path.join(LOG_DIR, "session_summary.csv")
EVENT_FILE = os.path.join(LOG_DIR, "event_log.csv")

os.makedirs(LOG_DIR, exist_ok=True)

SUMMARY_COLUMNS = [
    "Session_ID", "session_start", "session_end",
    "duration_seconds", "blink_count", "look_away_count",
    "eye_closure_warning_count", "center_gaze_seconds",
    "look_away_seconds", "unknown_gaze_seconds",
    "eyes_closed_seconds", "Face_Detected", "Calibrated",
    "Look_Away_Count", "Blink_Count", "Look_Away_Seconds",
    "Center_Seconds", "Unknown_Gaze_Seconds", "Last_Event"
]

FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473
RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (362, 263)


def angle_difference(angle, neutral):
    return (angle - neutral + 180) % 360 - 180


def eye_aspect_ratio(landmarks, indices, width, height):
    points = [
        np.array([
            landmarks[i].x * width,
            landmarks[i].y * height
        ], dtype=np.float32)
        for i in indices
    ]

    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal < 1:
        return 0.0

    return float(
        (vertical_1 + vertical_2) / (2.0 * horizontal)
    )


def get_head_pose(landmarks, width, height):
    image_points = np.array([
        [landmarks[i].x * width, landmarks[i].y * height]
        for i in POSE_LANDMARKS
    ], dtype=np.float64)

    focal = float(width)
    center = (width / 2.0, height / 2.0)

    camera_matrix = np.array([
        [focal, 0, center[0]],
        [0, focal, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    success, rotation_vector, _ = cv2.solvePnP(
        FACE_3D_MODEL,
        image_points,
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]

    return float(angles[1]), float(angles[0]), float(angles[2])


def get_eye_gaze(landmarks, iris_id, corners, width, height):
    iris = landmarks[iris_id]
    a = landmarks[corners[0]]
    b = landmarks[corners[1]]

    iris_x = iris.x * width
    iris_y = iris.y * height

    x1, x2 = a.x * width, b.x * width
    y1, y2 = a.y * height, b.y * height

    eye_width = abs(x2 - x1)

    if eye_width < 1:
        return None, None

    horizontal = (iris_x - min(x1, x2)) / eye_width
    vertical = (iris_y - ((y1 + y2) / 2.0)) / eye_width

    return float(horizontal), float(vertical)


def classify_gaze(x, y, neutral_x, neutral_y):
    if neutral_x is None or neutral_y is None:
        return "CALIBRATE GAZE"

    dx = x - neutral_x
    dy = y - neutral_y

    if dy < GAZE_VERTICAL_UP_DELTA:
        return "UP"
    if dy > GAZE_VERTICAL_DOWN_DELTA:
        return "DOWN"
    if dx < -GAZE_HORIZONTAL_DELTA:
        return "LEFT"
    if dx > GAZE_RIGHT_DELTA:
        return "RIGHT"

    return "CENTER"


def hide_face_region(frame, landmarks, width, height):
    if not HIDE_FACE or landmarks is None:
        return frame

    xs = [int(p.x * width) for p in landmarks]
    ys = [int(p.y * height) for p in landmarks]

    pad_x = int((max(xs) - min(xs)) * 0.15)
    pad_y = int((max(ys) - min(ys)) * 0.15)

    x1 = max(0, min(xs) - pad_x)
    y1 = max(0, min(ys) - pad_y)
    x2 = min(width, max(xs) + pad_x)
    y2 = min(height, max(ys) + pad_y)

    if x2 <= x1 or y2 <= y1:
        return frame

    roi = frame[y1:y2, x1:x2]
    small = cv2.resize(
        roi,
        (max(1, roi.shape[1] // 15),
         max(1, roi.shape[0] // 15))
    )

    frame[y1:y2, x1:x2] = cv2.resize(
        small,
        (roi.shape[1], roi.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    return frame


class NeuroVideoProcessor(VideoProcessorBase):

    def __init__(self):
        self.lock = threading.RLock()
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.reset_session()

    def reset_session(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.started_at = time.time()
        self.started_text = datetime.now().strftime(
            "%Y-%m-%dT%H:%M:%S"
        )

        self.blinks = 0
        self.lookaways = 0
        self.closure_warnings = 0

        self.center_seconds = 0.0
        self.lookaway_seconds = 0.0
        self.unknown_seconds = 0.0
        self.closed_seconds = 0.0

        self.last_analytics_time = time.time()
        self.last_gaze = "UNKNOWN"
        self.last_eye_status = "UNKNOWN"

        self.yaw_history = deque(maxlen=SMOOTHING_FRAMES)
        self.pitch_history = deque(maxlen=SMOOTHING_FRAMES)
        self.ear_history = deque(maxlen=SMOOTHING_FRAMES)
        self.gaze_x_history = deque(maxlen=SMOOTHING_FRAMES)
        self.gaze_y_history = deque(maxlen=SMOOTHING_FRAMES)
        self.gesture_history = deque(maxlen=SMOOTHING_FRAMES)
        self.gaze_history = deque(maxlen=SMOOTHING_FRAMES)

        self.calibration_yaw = []
        self.calibration_pitch = []
        self.calibration_x = []
        self.calibration_y = []

        self.neutral_yaw = None
        self.neutral_pitch = None
        self.neutral_x = None
        self.neutral_y = None

        self.closed_frames = 0
        self.eye_closed_start = None
        self.eye_closed_duration = 0.0
        self.lookaway_start = time.time()
        self.center_start = time.time()
        self.warning_active = False
        self.closure_warning_active = False

        self.gaze = "UNKNOWN"
        self.head = "NO FACE"
        self.eye_status = "UNKNOWN"
        self.message = "Position your face in the camera"
        self.face_detected = False
        self.calibrated = False
        self.last_event = "SESSION_STARTED"
        self.last_logged_head = None
        self.saved = False

        self.log_event("SESSION_STARTED", "Browser camera session started")

    def log_event(self, name, details=""):
        self.last_event = name

        try:
            exists = (
                os.path.exists(EVENT_FILE)
                and os.path.getsize(EVENT_FILE) > 0
            )

            with open(EVENT_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                if not exists:
                    writer.writerow([
                        "Session_ID", "Timestamp",
                        "Elapsed_seconds", "Event", "Details"
                    ])

                writer.writerow([
                    self.session_id,
                    datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                    round(time.time() - self.started_at, 2),
                    name,
                    details
                ])
        except (OSError, csv.Error):
            pass

    def calibrate_sample(self, pose, gx, gy):
        if self.calibrated:
            return

        if pose is not None:
            self.calibration_yaw.append(pose[0])
            self.calibration_pitch.append(pose[1])

        if gx is not None and gy is not None:
            self.calibration_x.append(gx)
            self.calibration_y.append(gy)

        self.message = (
            "Calibration: head {}/{} | gaze {}/{}".format(
                len(self.calibration_yaw),
                CALIBRATION_SAMPLES,
                len(self.calibration_x),
                CALIBRATION_SAMPLES
            )
        )

        if (
            len(self.calibration_yaw) >= CALIBRATION_SAMPLES
            and len(self.calibration_pitch) >= CALIBRATION_SAMPLES
            and len(self.calibration_x) >= CALIBRATION_SAMPLES
            and len(self.calibration_y) >= CALIBRATION_SAMPLES
        ):
            self.neutral_yaw = float(np.mean(self.calibration_yaw))
            self.neutral_pitch = float(np.mean(self.calibration_pitch))
            self.neutral_x = float(np.mean(self.calibration_x))
            self.neutral_y = float(np.mean(self.calibration_y))

            self.calibrated = True
            self.log_event(
                "CALIBRATION_COMPLETED",
                "Head and gaze baselines stored"
            )
            self.message = "Calibration complete"

    def reset_calibration(self):
        self.calibrated = False
        self.neutral_yaw = None
        self.neutral_pitch = None
        self.neutral_x = None
        self.neutral_y = None

        self.calibration_yaw.clear()
        self.calibration_pitch.clear()
        self.calibration_x.clear()
        self.calibration_y.clear()

        self.yaw_history.clear()
        self.pitch_history.clear()
        self.gaze_x_history.clear()
        self.gaze_y_history.clear()
        self.gaze_history.clear()
        self.gesture_history.clear()

        self.log_event("CALIBRATION_RESET", "Calibration reset by user")
        self.message = "Look straight; calibration restarted"

    def update_analytics(self, now):
        elapsed = min(1.0, max(0.0, now - self.last_analytics_time))

        if self.last_gaze == "CENTER":
            self.center_seconds += elapsed
        elif self.last_gaze in ("LEFT", "RIGHT", "UP", "DOWN"):
            self.lookaway_seconds += elapsed
        else:
            self.unknown_seconds += elapsed

        if self.last_eye_status == "CLOSED":
            self.closed_seconds += elapsed

        self.last_analytics_time = now

    def save_session(self):
        with self.lock:
            if self.saved:
                return
            self.saved = True

            end = time.time()
            self.update_analytics(end)

            row = {
                "Session_ID": self.session_id,
                "session_start": self.started_text,
                "session_end": datetime.fromtimestamp(end).strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
                "duration_seconds": round(end - self.started_at, 2),
                "blink_count": self.blinks,
                "look_away_count": self.lookaways,
                "eye_closure_warning_count": self.closure_warnings,
                "center_gaze_seconds": round(self.center_seconds, 2),
                "look_away_seconds": round(self.lookaway_seconds, 2),
                "unknown_gaze_seconds": round(self.unknown_seconds, 2),
                "eyes_closed_seconds": round(self.closed_seconds, 2),
                "Face_Detected": self.face_detected,
                "Calibrated": self.calibrated,
                "Look_Away_Count": self.lookaways,
                "Blink_Count": self.blinks,
                "Look_Away_Seconds": round(self.lookaway_seconds, 2),
                "Center_Seconds": round(self.center_seconds, 2),
                "Unknown_Gaze_Seconds": round(self.unknown_seconds, 2),
                "Last_Event": self.last_event
            }

            try:
                exists = (
                    os.path.exists(SUMMARY_FILE)
                    and os.path.getsize(SUMMARY_FILE) > 0
                )

                if exists:
                    with open(
                        SUMMARY_FILE, "r", newline="", encoding="utf-8"
                    ) as f:
                        header = next(csv.reader(f), [])

                    if header != SUMMARY_COLUMNS:
                        self.message = (
                            "Cannot save: session summary CSV header mismatch"
                        )
                        return

                with open(
                    SUMMARY_FILE, "a", newline="", encoding="utf-8"
                ) as f:
                    writer = csv.DictWriter(
                        f, fieldnames=SUMMARY_COLUMNS
                    )

                    if not exists:
                        writer.writeheader()

                    writer.writerow(row)

                self.log_event("SESSION_ENDED", "Browser camera session ended")
            except (OSError, csv.Error) as exc:
                self.message = "Could not save session: {}".format(exc)

    def recv(self, frame):
        with self.lock:
            image = frame.to_ndarray(format="bgr24")
            image = cv2.flip(image, 1)

            height, width = image.shape[:2]
            now = time.time()

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.mesh.process(rgb)

            self.face_detected = bool(results.multi_face_landmarks)

            if self.face_detected:
                landmarks = results.multi_face_landmarks[0].landmark

                pose = get_head_pose(landmarks, width, height)

                if pose is not None:
                    yaw, pitch, roll = pose
                    self.yaw_history.append(yaw)
                    self.pitch_history.append(pitch)

                    syaw = float(np.mean(self.yaw_history))
                    spitch = float(np.mean(self.pitch_history))

                    if self.calibrated:
                        dyaw = angle_difference(syaw, self.neutral_yaw)
                        dpitch = angle_difference(spitch, self.neutral_pitch)

                        if abs(dyaw) < YAW_THRESHOLD and abs(dpitch) < PITCH_THRESHOLD:
                            self.head = "CENTER"
                        elif dyaw > YAW_THRESHOLD:
                            self.head = "LEFT"
                        elif dyaw < -YAW_THRESHOLD:
                            self.head = "RIGHT"
                        elif dpitch > PITCH_THRESHOLD:
                            self.head = "DOWN"
                        elif dpitch < -PITCH_THRESHOLD:
                            self.head = "UP"
                    else:
                        self.head = "CALIBRATE"

                    cv2.putText(
                        image,
                        "Head: {} | Yaw: {:.1f} Pitch: {:.1f}".format(
                            self.head, syaw, spitch
                        ),
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 255), 2
                    )

                right_ear = eye_aspect_ratio(
                    landmarks, RIGHT_EYE, width, height
                )
                left_ear = eye_aspect_ratio(
                    landmarks, LEFT_EYE, width, height
                )
                ear = (right_ear + left_ear) / 2.0
                self.ear_history.append(ear)
                smooth_ear = float(np.mean(self.ear_history))

                if smooth_ear < BLINK_THRESHOLD:
                    self.closed_frames += 1
                    self.eye_status = "CLOSED"
                else:
                    if self.closed_frames >= MINIMUM_CLOSED_FRAMES:
                        self.blinks += 1
                        self.log_event("BLINK", "Blink detected")
                    self.closed_frames = 0
                    self.eye_status = "OPEN"

                rg = get_eye_gaze(
                    landmarks, RIGHT_IRIS_CENTER,
                    RIGHT_EYE_CORNERS, width, height
                )
                lg = get_eye_gaze(
                    landmarks, LEFT_IRIS_CENTER,
                    LEFT_EYE_CORNERS, width, height
                )

                valid = all(v is not None for v in rg + lg)

                if valid:
                    gx = (rg[0] + lg[0]) / 2.0
                    gy = (rg[1] + lg[1]) / 2.0

                    self.gaze_x_history.append(gx)
                    self.gaze_y_history.append(gy)

                    gx = float(np.mean(self.gaze_x_history))
                    gy = float(np.mean(self.gaze_y_history))

                    if not self.calibrated:
                        self.calibrate_sample(pose, gx, gy)

                    self.gaze_history.append(
                        classify_gaze(gx, gy, self.neutral_x, self.neutral_y)
                    )
                    self.gaze = max(
                        set(self.gaze_history),
                        key=self.gaze_history.count
                    )

                    if self.gaze == "CENTER":
                        self.lookaway_start = now
                        self.warning_active = False
                    elif self.gaze in ("LEFT", "RIGHT", "UP", "DOWN"):
                        if now - self.lookaway_start >= LOOKAWAY_WARNING_SECONDS:
                            if not self.warning_active:
                                self.lookaways += 1
                                self.log_event(
                                    "LOOK_AWAY_WARNING",
                                    "Gaze away from center"
                                )
                                self.warning_active = True

                    cv2.putText(
                        image,
                        "Gaze: {} | Blinks: {}".format(
                            self.gaze, self.blinks
                        ),
                        (15, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 0, 255), 2
                    )
                else:
                    self.gaze = "UNKNOWN"

                if self.eye_status == "CLOSED":
                    if self.eye_closed_start is None:
                        self.eye_closed_start = now
                    self.eye_closed_duration = now - self.eye_closed_start
                else:
                    self.eye_closed_start = None
                    self.eye_closed_duration = 0.0
                    self.closure_warning_active = False

                if (
                    self.eye_closed_duration >= LONG_EYE_CLOSURE_SECONDS
                    and not self.closure_warning_active
                ):
                    self.closure_warnings += 1
                    self.log_event(
                        "LONG_EYE_CLOSURE_WARNING",
                        "Eyes closed for a prolonged interval"
                    )
                    self.closure_warning_active = True

                if self.eye_closed_duration >= LONG_EYE_CLOSURE_SECONDS:
                    self.message = "Take a gentle break"
                elif not self.calibrated:
                    self.message = "Look straight; calibration in progress"
                elif self.gaze == "CENTER":
                    self.message = "Looking at screen"
                else:
                    self.message = "Gaze: " + self.gaze

                image = hide_face_region(
                    image, landmarks, width, height
                )
            else:
                self.gaze = "UNKNOWN"
                self.head = "NO FACE"
                self.eye_status = "UNKNOWN"
                self.message = "Face not detected - move into camera view"
                self.closed_frames = 0
                self.eye_closed_start = None
                self.eye_closed_duration = 0.0
                self.closure_warning_active = False

            self.last_gaze = self.gaze
            self.last_eye_status = self.eye_status
            self.update_analytics(now)

            cv2.putText(
                image,
                "Session: {:.1f}s | Center: {:.1f}s".format(
                    now - self.started_at, self.center_seconds
                ),
                (15, 100), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )
            cv2.putText(
                image, self.message[:75], (15, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

            return av.VideoFrame.from_ndarray(image, format="bgr24")
