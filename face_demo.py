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
HIDE_FACE = True
PITCH_THRESHOLD = 12

BLINK_THRESHOLD = 0.21
MINIMUM_CLOSED_FRAMES = 2
SMOOTHING_FRAMES = 5

GAZE_HORIZONTAL_DELTA = 0.07
GAZE_RIGHT_DELTA = 0.03
GAZE_VERTICAL_UP_DELTA = -0.025
GAZE_VERTICAL_DOWN_DELTA = 0.025

LOOKAWAY_WARNING_SECONDS = 1.0
LONG_EYE_CLOSURE_SECONDS = 2.0

# ==================================================
# SESSION ANALYTICS
# ==================================================

LOG_FOLDER = "session_logs"
os.makedirs(LOG_FOLDER, exist_ok=True)

SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SESSION_START = time.time()
SESSION_START_TEXT = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

SUMMARY_FILE = os.path.join(LOG_FOLDER, "session_summary.csv")
EVENT_FILE = os.path.join(LOG_FOLDER, "event_log.csv")

blink_count = 0
lookaway_counter = 0
eye_closure_warning_count = 0
# Head direction session counts
head_left_count = 0
head_right_count = 0
head_center_count = 0
head_up_count = 0
head_down_count = 0

last_logged_head_direction = None

center_gaze_seconds = 0.0
look_away_seconds = 0.0
unknown_gaze_seconds = 0.0
eyes_closed_seconds = 0.0

last_update_time = SESSION_START
last_gaze_for_analytics = "UNKNOWN"
last_eye_status = "UNKNOWN"



def log_event(event_name, details=""):
    global last_event
    last_event = event_name
    file_exists = (
        os.path.exists(EVENT_FILE)
        and os.path.getsize(EVENT_FILE) > 0
    )

    with open(EVENT_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "Session_ID", "Timestamp",
                "Elapsed_seconds", "Event", "Details"
            ])

        writer.writerow([
            SESSION_ID,
            datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            round(time.time() - SESSION_START, 2),
            event_name,
            details
        ])


def update_analytics(now, gaze, current_eye_status):
    global last_update_time
    global last_gaze_for_analytics, last_eye_status
    global center_gaze_seconds, look_away_seconds
    global unknown_gaze_seconds, eyes_closed_seconds

    elapsed = max(0.0, now - last_update_time)

    if last_gaze_for_analytics == "CENTER":
        center_gaze_seconds += elapsed
    elif last_gaze_for_analytics in ("LEFT", "RIGHT", "UP", "DOWN"):
        look_away_seconds += elapsed
    else:
        unknown_gaze_seconds += elapsed

    if last_eye_status == "CLOSED":
        eyes_closed_seconds += elapsed

    last_update_time = now
    last_gaze_for_analytics = gaze
    last_eye_status = current_eye_status




def save_session_summary(session_end):
    columns = [
        "Session_ID",
        "session_start",
        "session_end",
        "duration_seconds",
        "blink_count",
        "look_away_count",
        "eye_closure_warning_count",
        "center_gaze_seconds",
        "look_away_seconds",
        "unknown_gaze_seconds",
        "eyes_closed_seconds",
        "Face_Detected",
        "Calibrated",
        "Look_Away_Count",
        "Blink_Count",
        "Look_Away_Seconds",
        "Center_Seconds",
        "Unknown_Gaze_Seconds",
        "Last_Event"
    ]

    row = {
        "Session_ID": SESSION_ID,
        "session_start": SESSION_START_TEXT,
        "session_end": datetime.fromtimestamp(
            session_end
        ).strftime("%Y-%m-%dT%H:%M:%S"),
        "duration_seconds": round(
            session_end - SESSION_START, 2
        ),
        "blink_count": blink_count,
        "look_away_count": lookaway_counter,
        "eye_closure_warning_count": eye_closure_warning_count,
        "center_gaze_seconds": round(center_gaze_seconds, 2),
        "look_away_seconds": round(look_away_seconds, 2),
        "unknown_gaze_seconds": round(unknown_gaze_seconds, 2),
        "eyes_closed_seconds": round(eyes_closed_seconds, 2),
        "Face_Detected": True,
        "Calibrated": True,
        "Look_Away_Count": lookaway_counter,
        "Blink_Count": blink_count,
        "Look_Away_Seconds": round(look_away_seconds, 2),
        "Center_Seconds": round(center_gaze_seconds, 2),
        "Unknown_Gaze_Seconds": round(unknown_gaze_seconds, 2),
        "Last_Event": last_event
    }
    file_exists = (
        os.path.exists(SUMMARY_FILE)
        and os.path.getsize(SUMMARY_FILE) > 0
    )

    if file_exists:
        with open(
            SUMMARY_FILE, "r", newline="", encoding="utf-8"
        ) as f:
            existing_header = next(csv.reader(f), [])

        if existing_header != columns:
            print("ERROR: CSV header mismatch.")
            print("Session summary was not saved.")
            return

    with open(
        SUMMARY_FILE, "a", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=columns)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print("Session summary saved successfully.")

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

    horizontal = (iris_x - min(x1, x2)) / eye_width
    vertical = (iris_y - ((y1 + y2) / 2.0)) / eye_width

    return float(horizontal), float(vertical)



def classify_gaze(horizontal, vertical, neutral_x, neutral_y):
    if (
        horizontal is None or vertical is None
        or neutral_x is None or neutral_y is None
    ):
        return "CALIBRATE GAZE"

    dx = horizontal - neutral_x
    dy = vertical - neutral_y

    # Check horizontal and vertical movement
    if dy < GAZE_VERTICAL_UP_DELTA:
        return "UP"

    if dy > GAZE_VERTICAL_DOWN_DELTA:
        return "DOWN"

    if dx < -GAZE_HORIZONTAL_DELTA:
        return "LEFT"

    if dx > GAZE_RIGHT_DELTA:
        return "RIGHT"

    return "CENTER"



def most_common(history, default_value):
    if not history:
        return default_value

    return max(set(history), key=history.count)
def hide_face_region(frame, landmarks, width, height):
    if not HIDE_FACE or landmarks is None:
        return frame

    # Find the face area using MediaPipe landmarks
    xs = [int(point.x * width) for point in landmarks]
    ys = [int(point.y * height) for point in landmarks]

    padding_x = int((max(xs) - min(xs)) * 0.15)
    padding_y = int((max(ys) - min(ys)) * 0.15)

    x1 = max(0, min(xs) - padding_x)
    y1 = max(0, min(ys) - padding_y)
    x2 = min(width, max(xs) + padding_x)
    y2 = min(height, max(ys) + padding_y)

    if x2 > x1 and y2 > y1:
        roi = frame[y1:y2, x1:x2]

        # Strong pixelation to conceal facial details
        small = cv2.resize(
            roi,
            (max(1, roi.shape[1] // 15),
             max(1, roi.shape[0] // 15)),
            interpolation=cv2.INTER_LINEAR
        )

        pixelated = cv2.resize(
            small,
            (roi.shape[1], roi.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        frame[y1:y2, x1:x2] = pixelated

    return frame

def get_adaptive_message(gaze, eye_status, eye_closed_duration):
    if eye_closed_duration >= LONG_EYE_CLOSURE_SECONDS:
        return "Take a gentle break - eyes closed for a while"
    if eye_status == "CLOSED":
        return "Eyes closed"
    if gaze == "CENTER":
        return "Looking at screen - interface ready"
    if gaze == "LEFT":
        return "Looking left"
    if gaze == "RIGHT":
        return "Looking right"
    if gaze == "UP":
        return "Looking up"
    if gaze == "DOWN":
        return "Looking down"
    if gaze == "CALIBRATE GAZE":
        return "Look straight and calibrate gaze"

    return "Face or gaze not detected"


# ==================================================
# STATE
# ==================================================

yaw_history = deque(maxlen=SMOOTHING_FRAMES)
pitch_history = deque(maxlen=SMOOTHING_FRAMES)

neutral_yaw = None
neutral_pitch = None

calibration_yaw = []
calibration_pitch = []
calibration_gaze_x = []
calibration_gaze_y = []

neutral_gaze_x = None
neutral_gaze_y = None

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

center_start_time = time.time()
center_time = 0.0
highest_center_time = 0.0

eye_closed_start = None
eye_closed_duration = 0.0
eye_break_warning = False
eye_closure_warning_active = False

adaptive_message = "Interface Ready"

# ==================================================
# CAMERA
# ==================================================

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("ERROR: Could not open webcam. Try CAMERA_INDEX = 1.")
    face_mesh.close()
    raise SystemExit

log_event("SESSION_STARTED", "Session started")

print("-----------------------------------------")
print(" Neuro-Adaptive Interface")
print(" Look straight and press C 20 times")
print(" C = collect calibration sample")
print(" R = reset calibration")
print(" Q = quit and save session")
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

        now = time.time()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # STEP 2: HEAD POSE
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

                    cv2.putText(
                        frame, "Yaw delta: {:.1f}".format(yaw_delta),
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2
                    )
                    cv2.putText(
                        frame, "Pitch delta: {:.1f}".format(pitch_delta),
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2
                    )

                gesture_history.append(gesture)
                stable_gesture = most_common(gesture_history, "UNKNOWN")

                # Log head direction only when the stable direction changes
                if stable_gesture in ("LEFT", "RIGHT", "CENTER", "UP", "DOWN"):

                    if stable_gesture != last_logged_head_direction:

                        log_event(
                            "HEAD_DIRECTION",
                            stable_gesture
                        )

                        last_logged_head_direction = stable_gesture

                        if stable_gesture == "LEFT":
                            head_left_count += 1

                        elif stable_gesture == "RIGHT":
                            head_right_count += 1

                        elif stable_gesture == "CENTER":
                            head_center_count += 1

                        elif stable_gesture == "UP":
                            head_up_count += 1

                        elif stable_gesture == "DOWN":
                            head_down_count += 1

                cv2.putText(
                    frame,
                    "Yaw: {:.1f} Pitch: {:.1f} Roll: {:.1f}".format(
                        smooth_yaw, smooth_pitch, roll
                    ),
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2
                )

            # STEP 3: BLINK DETECTION
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
                    log_event("BLINK", "Blink detected")

                closed_frames = 0
                eye_status = "OPEN"

            for index in RIGHT_EYE + LEFT_EYE:
                px = int(landmarks[index].x * width)
                py = int(landmarks[index].y * height)
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

            cv2.putText(
                frame, "Eye: {}".format(eye_status), (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0) if eye_status == "OPEN" else (0, 0, 255), 2
            )
            cv2.putText(
                frame, "Blinks: {}".format(blink_count), (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2
            )
            cv2.putText(
                frame, "EAR: {:.3f}".format(smooth_ear), (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
            )

            # STEP 4: IRIS GAZE TRACKING
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

                for iris_id in range(468, min(478, len(landmarks))):
                    px = int(landmarks[iris_id].x * width)
                    py = int(landmarks[iris_id].y * height)
                    cv2.circle(frame, (px, py), 2, (0, 0, 255), -1)

                for iris_id in (RIGHT_IRIS_CENTER, LEFT_IRIS_CENTER):
                    px = int(landmarks[iris_id].x * width)
                    py = int(landmarks[iris_id].y * height)
                    cv2.circle(frame, (px, py), 5, (255, 0, 255), -1)

                cv2.putText(
                    frame, "Gaze: {}".format(stable_gaze), (20, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2
                )
                cv2.putText(
                    frame, "Gaze X:{:.3f} Y:{:.3f}".format(gaze_x, gaze_y),
                    (20, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2
                )
                cv2.putText(
                    frame, "Raw Y:{:.3f} dY:{:.3f}".format(
                        raw_gaze_y,
                        gaze_y - neutral_gaze_y if neutral_gaze_y is not None else 0.0
                    ),
                    (20, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 2
                )

                # STEP 5: LOOK-AWAY WARNING
                if stable_gaze == "CENTER":
                    if prev_gaze != "CENTER":
                        center_start_time = now

                    center_time = now - center_start_time
                    highest_center_time = max(highest_center_time, center_time)

                    warning_active = False
                    lookaway_start_time = now

                elif stable_gaze in ("LEFT", "RIGHT", "UP", "DOWN"):
                    center_time = 0.0

                    if prev_gaze not in ("LEFT", "RIGHT", "UP", "DOWN"):
                        lookaway_start_time = now
                        warning_active = False

                    if now - lookaway_start_time > LOOKAWAY_WARNING_SECONDS:
                        cv2.putText(
                            frame, "Look At The Screen!", (100, 100),
                            cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 3
                        )

                        if not warning_active:
                            lookaway_counter += 1
                            log_event(
                                "LOOK_AWAY_WARNING",
                                "Gaze away from center longer than warning delay"
                            )
                            warning_active = True
                else:
                    center_time = 0.0

                prev_gaze = stable_gaze

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

            # STEP 6: LONG EYE-CLOSURE WARNING
            if eye_status == "CLOSED":
                if eye_closed_start is None:
                    eye_closed_start = now
                eye_closed_duration = now - eye_closed_start
            else:
                eye_closed_start = None
                eye_closed_duration = 0.0

            eye_break_warning = eye_closed_duration >= LONG_EYE_CLOSURE_SECONDS

            if eye_break_warning and not eye_closure_warning_active:
                eye_closure_warning_count += 1
                log_event(
                    "LONG_EYE_CLOSURE_WARNING",
                    "Eyes closed for {:.2f} seconds".format(eye_closed_duration)
                )
                eye_closure_warning_active = True
            elif not eye_break_warning:
                eye_closure_warning_active = False

            adaptive_message = get_adaptive_message(
                stable_gaze, eye_status, eye_closed_duration
            )

            if eye_break_warning:
                message_color = (0, 0, 255)
            elif stable_gaze == "CENTER":
                message_color = (0, 255, 0)
            else:
                message_color = (0, 255, 255)

            cv2.rectangle(
                frame, (10, 365), (width - 10, 425), (30, 30, 30), -1
            )
            cv2.putText(
                frame, "ADAPTIVE RESPONSE:", (20, 388),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
            )
            cv2.putText(
                frame, adaptive_message[:65], (20, 413),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, message_color, 2
            )

        else:
            stable_gaze = "UNKNOWN"
            stable_gesture = "NO FACE"
            eye_status = "UNKNOWN"

            closed_frames = 0
            eye_closed_start = None
            eye_closed_duration = 0.0
            eye_break_warning = False
            eye_closure_warning_active = False

            ear_history.clear()
            gesture_history.clear()
            gaze_history.clear()
            gaze_x_history.clear()
            gaze_y_history.clear()

            adaptive_message = "Face not detected - position yourself in view"

            cv2.putText(
                frame, adaptive_message, (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
            )

        # Update timing analytics every frame
        update_analytics(now, stable_gaze, eye_status)

        # Live analytics overlay
        cv2.putText(
            frame, "Session: {:.1f}s".format(now - SESSION_START),
            (width - 230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 2
        )
        cv2.putText(
            frame, "Center: {:.1f}s".format(center_gaze_seconds),
            (width - 230, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 255, 0), 1
        )
        cv2.putText(
            frame, "Look-away: {:.1f}s".format(look_away_seconds),
            (width - 230, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 255), 1
        )
        cv2.putText(
            frame, "Unknown: {:.1f}s".format(unknown_gaze_seconds),
            (width - 230, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (200, 200, 200), 1
        )

        # DISPLAY
        cv2.putText(
            frame, "Head: {}".format(stable_gesture), (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        cv2.putText(
            frame, "C: Calibrate | R: Reset | Q: Quit",
            (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 2
        )
        # Hide the face only in the displayed camera frame
        if results.multi_face_landmarks:
            display_landmarks = results.multi_face_landmarks[0].landmark
            frame = hide_face_region(
                frame, display_landmarks, width, height
            )

        cv2.imshow("Neuro-Adaptive Interface", frame)
        key = cv2.waitKey(1) & 0xFF

        # KEYBOARD CONTROLS
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

                if (
                    right_sample[0] is not None and right_sample[1] is not None
                    and left_sample[0] is not None and left_sample[1] is not None
                ):
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

                if (
                    len(calibration_yaw) >= CALIBRATION_SAMPLES
                    and len(calibration_pitch) >= CALIBRATION_SAMPLES
                    and len(calibration_gaze_x) >= CALIBRATION_SAMPLES
                    and len(calibration_gaze_y) >= CALIBRATION_SAMPLES
                ):
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

                    log_event("CALIBRATION_COMPLETE", "Neutral values saved")
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
            lookaway_counter = 0

            eye_closed_start = None
            eye_closed_duration = 0.0
            eye_break_warning = False
            eye_closure_warning_active = False

            print("Calibration reset. Look straight and press C 20 times.")
            log_event("CALIBRATION_RESET", "User reset calibration")

        elif key == ord("q"):
            log_event("SESSION_ENDED", "User pressed Q")
            break

finally:
    final_time = time.time()
    update_analytics(final_time, stable_gaze, eye_status)

    camera.release()
    face_mesh.close()
    cv2.destroyAllWindows()

    save_session_summary(final_time)

    print("Program closed.")
    print("Session summary saved to:", SUMMARY_FILE)
    print("Event log saved to:", EVENT_FILE)
    print("Session duration: {:.2f}s".format(final_time - SESSION_START))
    print("Blinks:", blink_count)
    print("Look-away warnings:", lookaway_counter)
    print("Eye-closure warnings:", eye_closure_warning_count)
    print("Center gaze seconds:", round(center_gaze_seconds, 2))
    print("Look-away seconds:", round(look_away_seconds, 2))
    print("Unknown gaze seconds:", round(unknown_gaze_seconds, 2))
    print("Eyes closed seconds:", round(eyes_closed_seconds, 2))
    print("Head LEFT events:", head_left_count)
    print("Head RIGHT events:", head_right_count)
    print("Head CENTER events:", head_center_count)
    print("Head UP events:", head_up_count)
    print("Head DOWN events:", head_down_count)