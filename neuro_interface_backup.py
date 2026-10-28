import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from datetime import datetime
from collections import deque

# ==================================================
# SETTINGS
# ==================================================
CAMERA_INDEX = 0
CALIBRATION_SAMPLES = 20

YAW_THRESHOLD = 15
PITCH_THRESHOLD = 12

BLINK_THRESHOLD = 0.21
MINIMUM_CLOSED_FRAMES = 2
SMOOTHING_FRAMES = 5

# Gaze thresholds are relative to your calibrated straight-ahead gaze.
# If gaze directions are reversed on your webcam, see the notes after the code.
GAZE_HORIZONTAL_DELTA = 0.07
GAZE_VERTICAL_UP_DELTA = -0.025
GAZE_VERTICAL_DOWN_DELTA = 0.025

LOOKAWAY_WARNING_SECONDS = 0.3

# ==================================================
# MEDIAPIPE
# ==================================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==================================================
# HEAD POSE LANDMARKS
# ==================================================
FACE_3D_MODEL = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

# ==================================================
# EYE LANDMARKS
# ==================================================
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473

RIGHT_EYE_CORNERS = (33, 133)
LEFT_EYE_CORNERS = (362, 263)

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def angle_difference(angle, neutral):
    return (angle - neutral + 180) % 360 - 180


def eye_aspect_ratio(landmarks, eye_indices, width, height):
    points = []
    for index in eye_indices:
        point = landmarks[index]
        points.append(np.array(
            [point.x * width, point.y * height],
            dtype=np.float32
        ))

    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal < 1:
        return 0.0

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def get_head_pose(landmarks, width, height):
    image_points = []
    for index in POSE_LANDMARKS:
        point = landmarks[index]
        image_points.append([point.x * width, point.y * height])

    image_points = np.array(image_points, dtype=np.float64)
    focal_length = width
    center = (width / 2.0, height / 2.0)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1), dtype=np.float64)
    success, rotation_vector, _ = cv2.solvePnP(
        FACE_3D_MODEL,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]
    return float(angles[1]), float(angles[0]), float(angles[2])


def get_eye_gaze(landmarks, iris_center_id, eye_corners, width, height):
    """Return iris position normalized by eye width; y increases downward."""
    iris = landmarks[iris_center_id]
    iris_x = iris.x * width
    iris_y = iris.y * height

    corner_a = landmarks[eye_corners[0]]
    corner_b = landmarks[eye_corners[1]]
    x1, x2 = corner_a.x * width, corner_b.x * width
    y1, y2 = corner_a.y * height, corner_b.y * height

    eye_width = abs(x2 - x1)
    if eye_width < 1:
        return None, None

    # Horizontal position runs from approximately 0 to 1 across the eye.
    horizontal = (iris_x - min(x1, x2)) / eye_width

    # Vertical position is measured relative to the line joining eye corners.
    vertical = (iris_y - ((y1 + y2) / 2.0)) / eye_width
    return float(horizontal), float(vertical)


def classify_gaze(horizontal, vertical, neutral_x, neutral_y):
    if (horizontal is None or vertical is None
            or neutral_x is None or neutral_y is None):
        return "CALIBRATE GAZE"

    dx = horizontal - neutral_x
    dy = vertical - neutral_y

    # Vertical first, so looking down/up is not mislabelled left/right.
    if dy < GAZE_VERTICAL_UP_DELTA:
        return "UP"
    if dy > GAZE_VERTICAL_DOWN_DELTA:
        return "DOWN"

    if dx < -GAZE_HORIZONTAL_DELTA:
        return "LEFT"
    if dx > GAZE_HORIZONTAL_DELTA:
        return "RIGHT"
    return "CENTER"


def most_common(history, default_value):
    if not history:
        return default_value
    return max(set(history), key=history.count)


# ==================================================
# SESSION ANALYTICS AND CSV REPORT
# ==================================================
REPORT_FOLDER = "session_reports"
LONG_EYE_CLOSURE_SECONDS = 2.0

class SessionAnalytics:
    def __init__(self):
        self.start_time = time.time()
        self.last_update = self.start_time
        self.blink_count = 0
        self.look_away_count = 0
        self.eye_warning_count = 0
        self.center_seconds = 0.0
        self.look_away_seconds = 0.0
        self.unknown_seconds = 0.0
        self.eye_closed_seconds = 0.0
        self.events = []
        self.log_event("SESSION_STARTED", "Session started")

    def log_event(self, event_name, details=""):
        self.events.append([
            datetime.now().isoformat(timespec="seconds"),
            round(time.time() - self.start_time, 2),
            event_name, details
        ])

    def update_time(self, gaze, eye_status):
        now = time.time()
        elapsed = max(0.0, now - self.last_update)
        self.last_update = now
        if gaze == "CENTER":
            self.center_seconds += elapsed
        elif gaze in ("LEFT", "RIGHT", "UP", "DOWN"):
            self.look_away_seconds += elapsed
        else:
            self.unknown_seconds += elapsed
        if eye_status == "CLOSED":
            self.eye_closed_seconds += elapsed

    def save_report(self):
        os.makedirs(REPORT_FOLDER, exist_ok=True)
        end_time = datetime.now()
        duration = time.time() - self.start_time
        filename = "session_report_" + end_time.strftime("%Y%m%d_%H%M%S") + ".csv"
        path = os.path.join(REPORT_FOLDER, filename)
        summary = [
            ("session_start", datetime.fromtimestamp(self.start_time).isoformat(timespec="seconds")),
            ("session_end", end_time.isoformat(timespec="seconds")),
            ("duration_seconds", round(duration, 2)),
            ("blink_count", self.blink_count),
            ("look_away_count", self.look_away_count),
            ("eye_closure_warning_count", self.eye_warning_count),
            ("center_gaze_seconds", round(self.center_seconds, 2)),
            ("look_away_seconds", round(self.look_away_seconds, 2)),
            ("unknown_gaze_seconds", round(self.unknown_seconds, 2)),
            ("eyes_closed_seconds", round(self.eye_closed_seconds, 2))
        ]
        with open(path, "w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["SESSION SUMMARY"])
            writer.writerow(["Metric", "Value"])
            writer.writerows(summary)
            writer.writerow([])
            writer.writerow(["EVENT LOG"])
            writer.writerow(["Timestamp", "Elapsed seconds", "Event", "Details"])
            writer.writerows(self.events)
        return path, summary

# ==================================================
# STATE
# ==================================================
yaw_history = deque(maxlen=SMOOTHING_FRAMES)
pitch_history = deque(maxlen=SMOOTHING_FRAMES)
neutral_yaw = None
neutral_pitch = None
calibration_yaw = []
calibration_pitch = []

# Gaze calibration uses the same C key: look straight at the screen
# and press C 20 times. Samples are collected only while C is pressed.
calibration_gaze_x = []
calibration_gaze_y = []
neutral_gaze_x = None
neutral_gaze_y = None

blink_count = 0
analytics = SessionAnalytics()
closed_frames = 0
eye_status = "UNKNOWN"
ear_history = deque(maxlen=SMOOTHING_FRAMES)

gesture_history = deque(maxlen=SMOOTHING_FRAMES)
stable_gesture = "NO FACE"

gaze_history = deque(maxlen=SMOOTHING_FRAMES)
gaze_x_history = deque(maxlen=SMOOTHING_FRAMES)
gaze_y_history = deque(maxlen=SMOOTHING_FRAMES)
stable_gaze = "UNKNOWN"

prev_gaze = "CENTER"
lookaway_start_time = time.time()
warning_active = False
lookaway_counter = 0

center_start_time = time.time()
center_time = 0.0
highest_center_time = 0.0
eye_closed_start = None
eye_warning_active = False

# ==================================================
# CAMERA
# ==================================================
camera = cv2.VideoCapture(CAMERA_INDEX)
if not camera.isOpened():
    print("ERROR: Could not open webcam. Try CAMERA_INDEX = 1.")
    face_mesh.close()
    raise SystemExit

print("-----------------------------------------")
print(" Neuro-Adaptive Interface")
print(" Look straight at the screen, then press C 20 times")
print(" C = collect calibration sample")
print(" R = reset head/gaze calibration")
print(" Q = quit")
print("-----------------------------------------")


# ==================================================
# MAIN LOOP
# ==================================================
try:
    while True:
        ret, frame = camera.read()
        if not ret:
            print("ERROR: Could not read webcam frame.")
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # --------------------------------------------------
            # STEP 2: HEAD POSE
            # --------------------------------------------------
            pose = get_head_pose(landmarks, width, height)
            if pose is not None:
                yaw, pitch, roll = pose
                yaw_history.append(yaw)
                pitch_history.append(pitch)
                smooth_yaw = float(np.mean(yaw_history))
                smooth_pitch = float(np.mean(pitch_history))

                for index in POSE_LANDMARKS:
                    px = int(landmarks[index].x * width)
                    py = int(landmarks[index].y * height)
                    cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)

                if neutral_yaw is None or neutral_pitch is None:
                    gesture = "PRESS C TO CALIBRATE"
                else:
                    yaw_delta = angle_difference(smooth_yaw, neutral_yaw)
                    pitch_delta = angle_difference(smooth_pitch, neutral_pitch)

                    if abs(yaw_delta) < YAW_THRESHOLD and abs(pitch_delta) < PITCH_THRESHOLD:
                        gesture = "CENTER"
                    elif yaw_delta > YAW_THRESHOLD:
                        gesture = "LEFT"
                    elif yaw_delta < -YAW_THRESHOLD:
                        gesture = "RIGHT"
                    elif pitch_delta > PITCH_THRESHOLD:
                        gesture = "DOWN"
                    elif pitch_delta < -PITCH_THRESHOLD:
                        gesture = "UP"
                    else:
                        gesture = "CENTER"

                    cv2.putText(frame, "Yaw delta: {:.1f}".format(yaw_delta),
                                (20, 90), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 255, 255), 2)
                    cv2.putText(frame, "Pitch delta: {:.1f}".format(pitch_delta),
                                (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 255, 255), 2)

                gesture_history.append(gesture)
                stable_gesture = most_common(gesture_history, "UNKNOWN")

                cv2.putText(
                    frame,
                    "Yaw: {:.1f} Pitch: {:.1f} Roll: {:.1f}".format(
                        smooth_yaw, smooth_pitch, roll
                    ),
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2
                )

            # --------------------------------------------------
            # STEP 3: BLINK DETECTION
            # --------------------------------------------------
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, width, height)
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, width, height)
            ear = (right_ear + left_ear) / 2.0
            ear_history.append(ear)
            smooth_ear = float(np.mean(ear_history))

            if smooth_ear < BLINK_THRESHOLD:
                closed_frames += 1
                eye_status = "CLOSED"
            else:
                if closed_frames >= MINIMUM_CLOSED_FRAMES:
                    blink_count += 1
                    analytics.blink_count = blink_count
                    analytics.log_event("BLINK", "Blink detected")
                closed_frames = 0
                eye_status = "OPEN"

            for index in RIGHT_EYE + LEFT_EYE:
                px = int(landmarks[index].x * width)
                py = int(landmarks[index].y * height)
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

            cv2.putText(frame, "Eye: {}".format(eye_status), (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 0) if eye_status == "OPEN" else (0, 0, 255), 2)
            cv2.putText(frame, "Blinks: {}".format(blink_count), (20, 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.putText(frame, "EAR: {:.3f}".format(smooth_ear), (20, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # --------------------------------------------------
            # STEP 4: IRIS GAZE TRACKING
            # --------------------------------------------------
            right_gaze = get_eye_gaze(
                landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_CORNERS, width, height
            )
            left_gaze = get_eye_gaze(
                landmarks, LEFT_IRIS_CENTER, LEFT_EYE_CORNERS, width, height
            )

            valid_gaze = (
                right_gaze[0] is not None and right_gaze[1] is not None
                and left_gaze[0] is not None and left_gaze[1] is not None
            )

            if valid_gaze:
                raw_gaze_x = (right_gaze[0] + left_gaze[0]) / 2.0
                raw_gaze_y = (right_gaze[1] + left_gaze[1]) / 2.0

                gaze_x_history.append(raw_gaze_x)
                gaze_y_history.append(raw_gaze_y)
                gaze_x = float(np.mean(gaze_x_history))
                gaze_y = float(np.mean(gaze_y_history))

                gaze_label = classify_gaze(
                    gaze_x, gaze_y, neutral_gaze_x, neutral_gaze_y
                )
                gaze_history.append(gaze_label)
                stable_gaze = most_common(gaze_history, "UNKNOWN")

                # Draw iris landmarks and centers.
                for iris_id in range(468, min(478, len(landmarks))):
                    px = int(landmarks[iris_id].x * width)
                    py = int(landmarks[iris_id].y * height)
                    cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)

                for iris_id in (RIGHT_IRIS_CENTER, LEFT_IRIS_CENTER):
                    px = int(landmarks[iris_id].x * width)
                    py = int(landmarks[iris_id].y * height)
                    cv2.circle(frame, (px, py), 5, (255, 0, 255), -1)

                cv2.putText(frame, "Gaze: {}".format(stable_gaze), (20, 270),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                cv2.putText(
                    frame,
                    "Gaze X:{:.3f} Y:{:.3f}".format(gaze_x, gaze_y),
                    (20, 305), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2
                )
                cv2.putText(
                    frame,
                    "Raw Y:{:.3f} dY:{:.3f}".format(
                        raw_gaze_y,
                        (gaze_y - neutral_gaze_y) if neutral_gaze_y is not None else 0.0
                    ),
                    (20, 335), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2
                )

                # --------------------------------------------------
                # STEP 5: LOOK-AWAY WARNING
                # --------------------------------------------------
                now = time.time()
                if stable_gaze == "CENTER":
                    if prev_gaze != "CENTER":
                        center_start_time = now
                    center_time = now - center_start_time
                    highest_center_time = max(highest_center_time, center_time)
                    warning_active = False
                    lookaway_start_time = now
                else:
                    center_time = 0.0
                    if prev_gaze == "CENTER":
                        lookaway_start_time = now
                        warning_active = False

                    if now - lookaway_start_time > LOOKAWAY_WARNING_SECONDS:
                        cv2.putText(frame, "Look At The Screen!", (100, 100),
                                    cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 3)
                        if not warning_active:
                            lookaway_counter += 1
                            analytics.look_away_count = lookaway_counter
                            analytics.log_event(
                                "LOOK_AWAY_WARNING",
                                "Gaze away from center for longer than warning delay"
                            )
                            warning_active = True

                prev_gaze = stable_gaze

                # Track prolonged eye closure for the session report.
                if eye_status == "CLOSED":
                    if eye_closed_start is None:
                        eye_closed_start = now
                    if (now - eye_closed_start >= LONG_EYE_CLOSURE_SECONDS
                            and not eye_warning_active):
                        analytics.eye_warning_count += 1
                        analytics.log_event(
                            "EYE_CLOSURE_WARNING",
                            "Eyes closed for at least 2 seconds"
                        )
                        eye_warning_active = True
                else:
                    if eye_warning_active:
                        analytics.log_event(
                            "EYE_CLOSURE_WARNING_ENDED",
                            "Eyes reopened"
                        )
                    eye_closed_start = None
                    eye_warning_active = False

                cv2.putText(
                    frame, "Center Time: {:.2f}s".format(center_time),
                    (width - 250, height - 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1
                )
                cv2.putText(
                    frame, "Max Center: {:.2f}s".format(highest_center_time),
                    (width - 250, height - 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 1
                )
                cv2.putText(
                    frame, "Look Away Count: {}".format(lookaway_counter),
                    (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2
                )
            else:
                stable_gaze = "UNKNOWN"
                gaze_history.clear()
                gaze_x_history.clear()
                gaze_y_history.clear()

        else:

            # Face is not detected

            eye_status = "UNKNOWN"

            # Clear previous tracking data

            ear_history.clear()

            gaze_history.clear()

            gaze_x_history.clear()

            gaze_y_history.clear()

            closed_frames = 0

            # Reset displayed tracking states

            gesture_history.clear()

            stable_gesture = "NO FACE"

            stable_gaze = "UNKNOWN"
        # Stop an active eye-closure warning
            if eye_warning_active:
                analytics.log_event(
                    "EYE_CLOSURE_WARNING_ENDED",
                    "Face lost; eye status is unknown"
                )
            eye_closed_start = None
            eye_warning_active = False

        # Update accumulated session times on every frame.
        analytics.update_time(stable_gaze, eye_status)

        # --------------------------------------------------
        # DISPLAY
        # --------------------------------------------------
        cv2.putText(frame, "Head: {}".format(stable_gesture), (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "C: Calibrate | R: Reset | Q: Quit",
                    (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)

        cv2.imshow("Neuro-Adaptive Interface", frame)
        key = cv2.waitKey(1) & 0xFF

        # --------------------------------------------------
        # KEYBOARD CONTROLS
        # --------------------------------------------------
        if key == ord("c"):
            if results.multi_face_landmarks:
                current_landmarks = results.multi_face_landmarks[0].landmark
                pose = get_head_pose(current_landmarks, width, height)
                right_sample = get_eye_gaze(
                    current_landmarks, RIGHT_IRIS_CENTER,
                    RIGHT_EYE_CORNERS, width, height
                )
                left_sample = get_eye_gaze(
                    current_landmarks, LEFT_IRIS_CENTER,
                    LEFT_EYE_CORNERS, width, height
                )

                if pose is not None:
                    calibration_yaw.append(pose[0])
                    calibration_pitch.append(pose[1])

                if (right_sample[0] is not None and right_sample[1] is not None
                        and left_sample[0] is not None and left_sample[1] is not None):
                    calibration_gaze_x.append(
                        (right_sample[0] + left_sample[0]) / 2.0
                    )
                    calibration_gaze_y.append(
                        (right_sample[1] + left_sample[1]) / 2.0
                    )

                print(
                    "Calibration samples - head: {}/{}, gaze: {}/{}".format(
                        len(calibration_yaw), CALIBRATION_SAMPLES,
                        len(calibration_gaze_x), CALIBRATION_SAMPLES
                    )
                )

                if (len(calibration_yaw) >= CALIBRATION_SAMPLES
                        and len(calibration_pitch) >= CALIBRATION_SAMPLES
                        and len(calibration_gaze_x) >= CALIBRATION_SAMPLES
                        and len(calibration_gaze_y) >= CALIBRATION_SAMPLES):
                    neutral_yaw = float(np.mean(calibration_yaw))
                    neutral_pitch = float(np.mean(calibration_pitch))
                    neutral_gaze_x = float(np.mean(calibration_gaze_x))
                    neutral_gaze_y = float(np.mean(calibration_gaze_y))

                    print("Calibration complete.")
                    print("Neutral yaw:", neutral_yaw)
                    print("Neutral pitch:", neutral_pitch)
                    print("Neutral gaze X:", neutral_gaze_x)
                    print("Neutral gaze Y:", neutral_gaze_y)

                    calibration_yaw.clear()
                    calibration_pitch.clear()
                    calibration_gaze_x.clear()
                    calibration_gaze_y.clear()
                    gaze_history.clear()
                    gaze_x_history.clear()
                    gaze_y_history.clear()
            else:
                print("Face not detected. Keep your face visible and press C again.")

        elif key == ord("r"):
            neutral_yaw = None
            neutral_pitch = None
            neutral_gaze_x = None
            neutral_gaze_y = None

            calibration_yaw.clear()
            calibration_pitch.clear()
            calibration_gaze_x.clear()
            calibration_gaze_y.clear()

            yaw_history.clear()
            pitch_history.clear()
            gesture_history.clear()
            gaze_history.clear()
            gaze_x_history.clear()
            gaze_y_history.clear()

            prev_gaze = "CENTER"
            lookaway_start_time = time.time()
            center_start_time = time.time()
            center_time = 0.0
            highest_center_time = 0.0
            warning_active = False
            eye_closed_start = None
            eye_warning_active = False
            analytics.log_event("CALIBRATION_RESET", "Calibration reset by user")
            print("Calibration reset. Look straight at the screen and press C 20 times.")

        elif key == ord("q"):
            analytics.log_event("SESSION_ENDED", "User pressed Q")
            break

finally:
    camera.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    print("Program closed.")

    report_path, summary = analytics.save_report()
    print("\n===== SESSION ANALYTICS SUMMARY =====")
    for metric, value in summary:
        print("{}: {}".format(metric.replace("_", " ").title(), value))
    print("\nCSV report saved to:")
    print(os.path.abspath(report_path))
