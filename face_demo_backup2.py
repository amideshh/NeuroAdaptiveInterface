import cv2
import mediapipe as mp
import numpy as np
import time
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

GAZE_HORIZONTAL_DELTA = 0.07
GAZE_RIGHT_DELTA = 0.03
GAZE_VERTICAL_UP_DELTA = -0.025
GAZE_VERTICAL_DOWN_DELTA = 0.025

LOOKAWAY_WARNING_SECONDS = 0.3
LONG_EYE_CLOSURE_SECONDS = 2.0

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

    return float(
        (vertical_1 + vertical_2) / (2.0 * horizontal)
    )


def get_head_pose(landmarks, width, height):
    image_points = []

    for index in POSE_LANDMARKS:
        point = landmarks[index]
        image_points.append([
            point.x * width,
            point.y * height
        ])

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

    return (
        float(angles[1]),
        float(angles[0]),
        float(angles[2])
    )


def get_eye_gaze(
    landmarks,
    iris_center_id,
    eye_corners,
    width,
    height
):
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

    vertical = (
        iris_y - ((y1 + y2) / 2.0)
    ) / eye_width

    return float(horizontal), float(vertical)


def classify_gaze(horizontal, vertical, neutral_x, neutral_y):
    if (
        horizontal is None
        or vertical is None
        or neutral_x is None
        or neutral_y is None
    ):
        return "CALIBRATE GAZE"

    dx = horizontal - neutral_x
    dy = vertical - neutral_y

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


# ==================================================
# STEP 6: NEURO-ADAPTIVE RESPONSE
# ==================================================

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

blink_count = 0
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

# Neuro-adaptive state
eye_closed_start = None
eye_closed_duration = 0.0
eye_break_warning = False
adaptive_message = "Interface Ready"

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
print(" Look straight and press C 20 times")
print(" C = collect calibration sample")
print(" R = reset calibration")
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

        now = time.time()

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # ------------------------------------------
            # STEP 2: HEAD POSE
            # ------------------------------------------

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

                    cv2.circle(
                        frame, (px, py), 4, (0, 255, 255), -1
                    )

                if neutral_yaw is None or neutral_pitch is None:
                    gesture = "PRESS C TO CALIBRATE"
                else:
                    yaw_delta = angle_difference(
                        smooth_yaw, neutral_yaw
                    )

                    pitch_delta = angle_difference(
                        smooth_pitch, neutral_pitch
                    )

                    if (
                        abs(yaw_delta) < YAW_THRESHOLD
                        and abs(pitch_delta) < PITCH_THRESHOLD
                    ):
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
                        frame,
                        "Yaw delta: {:.1f}".format(yaw_delta),
                        (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2
                    )

                    cv2.putText(
                        frame,
                        "Pitch delta: {:.1f}".format(pitch_delta),
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 2
                    )

                gesture_history.append(gesture)
                stable_gesture = most_common(
                    gesture_history, "UNKNOWN"
                )

                cv2.putText(
                    frame,
                    "Yaw: {:.1f} Pitch: {:.1f} Roll: {:.1f}".format(
                        smooth_yaw, smooth_pitch, roll
                    ),
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2
                )

            # ------------------------------------------
            # STEP 3: BLINK DETECTION
            # ------------------------------------------

            right_ear = eye_aspect_ratio(
                landmarks, RIGHT_EYE, width, height
            )

            left_ear = eye_aspect_ratio(
                landmarks, LEFT_EYE, width, height
            )

            ear = (right_ear + left_ear) / 2.0

            ear_history.append(ear)
            smooth_ear = float(np.mean(ear_history))

            if smooth_ear < BLINK_THRESHOLD:
                closed_frames += 1
                eye_status = "CLOSED"
            else:
                if closed_frames >= MINIMUM_CLOSED_FRAMES:
                    blink_count += 1

                closed_frames = 0
                eye_status = "OPEN"

            for index in RIGHT_EYE + LEFT_EYE:
                px = int(landmarks[index].x * width)
                py = int(landmarks[index].y * height)

                cv2.circle(
                    frame, (px, py), 2, (0, 255, 0), -1
                )

            cv2.putText(
                frame,
                "Eye: {}".format(eye_status),
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if eye_status == "OPEN"
                else (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "Blinks: {}".format(blink_count),
                (20, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 0), 2
            )

            cv2.putText(
                frame,
                "EAR: {:.3f}".format(smooth_ear),
                (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2
            )

            # ------------------------------------------
            # STEP 4: IRIS GAZE TRACKING
            # ------------------------------------------

            right_gaze = get_eye_gaze(
                landmarks,
                RIGHT_IRIS_CENTER,
                RIGHT_EYE_CORNERS,
                width,
                height
            )

            left_gaze = get_eye_gaze(
                landmarks,
                LEFT_IRIS_CENTER,
                LEFT_EYE_CORNERS,
                width,
                height
            )

            valid_gaze = (
                right_gaze[0] is not None
                and right_gaze[1] is not None
                and left_gaze[0] is not None
                and left_gaze[1] is not None
            )

            if valid_gaze:
                raw_gaze_x = (
                    right_gaze[0] + left_gaze[0]
                ) / 2.0

                raw_gaze_y = (
                    right_gaze[1] + left_gaze[1]
                ) / 2.0

                gaze_x_history.append(raw_gaze_x)
                gaze_y_history.append(raw_gaze_y)

                gaze_x = float(np.mean(gaze_x_history))
                gaze_y = float(np.mean(gaze_y_history))

                gaze_label = classify_gaze(
                    gaze_x,
                    gaze_y,
                    neutral_gaze_x,
                    neutral_gaze_y
                )

                gaze_history.append(gaze_label)

                stable_gaze = most_common(
                    gaze_history, "UNKNOWN"
                )

                for iris_id in range(
                    468, min(478, len(landmarks))
                ):
                    px = int(landmarks[iris_id].x * width)
                    py = int(landmarks[iris_id].y * height)

                    cv2.circle(
                        frame, (px, py), 2, (0, 0, 255), -1
                    )

                for iris_id in (
                    RIGHT_IRIS_CENTER,
                    LEFT_IRIS_CENTER
                ):
                    px = int(landmarks[iris_id].x * width)
                    py = int(landmarks[iris_id].y * height)

                    cv2.circle(
                        frame, (px, py), 5, (255, 0, 255), -1
                    )

                cv2.putText(
                    frame,
                    "Gaze: {}".format(stable_gaze),
                    (20, 270),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 0, 255), 2
                )

                cv2.putText(
                    frame,
                    "Gaze X:{:.3f} Y:{:.3f}".format(
                        gaze_x, gaze_y
                    ),
                    (20, 305),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2
                )

                cv2.putText(
                    frame,
                    "Raw Y:{:.3f} dY:{:.3f}".format(
                        raw_gaze_y,
                        gaze_y - neutral_gaze_y
                        if neutral_gaze_y is not None else 0.0
                    ),
                    (20, 335),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2
                )

                # --------------------------------------
                # STEP 5: LOOK-AWAY WARNING
                # --------------------------------------

                if stable_gaze == "CENTER":
                    if prev_gaze != "CENTER":
                        center_start_time = now

                    center_time = now - center_start_time
                    highest_center_time = max(
                        highest_center_time, center_time
                    )

                    warning_active = False
                    lookaway_start_time = now

                elif stable_gaze in (
                    "LEFT", "RIGHT", "UP", "DOWN"
                ):
                    center_time = 0.0

                    if prev_gaze == "CENTER":
                        lookaway_start_time = now
                        warning_active = False

                    if (
                        now - lookaway_start_time
                        > LOOKAWAY_WARNING_SECONDS
                    ):
                        cv2.putText(
                            frame,
                            "Look At The Screen!",
                            (100, 100),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1.0, (0, 0, 255), 3
                        )

                        if not warning_active:
                            lookaway_counter += 1
                            warning_active = True

                else:
                    center_time = 0.0
                    warning_active = False
                    lookaway_start_time = now

                prev_gaze = stable_gaze

                cv2.putText(
                    frame,
                    "Center Time: {:.2f}s".format(center_time),
                    (width - 250, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1
                )

                cv2.putText(
                    frame,
                    "Max Center: {:.2f}s".format(highest_center_time),
                    (width - 250, height - 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 1
                )

                cv2.putText(
                    frame,
                    "Look Away Count: {}".format(lookaway_counter),
                    (20, height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2
                )

            else:
                stable_gaze = "UNKNOWN"
                gaze_history.clear()
                gaze_x_history.clear()
                gaze_y_history.clear()

            # ------------------------------------------
            # STEP 6: NEURO-ADAPTIVE RESPONSE
            # ------------------------------------------

            if eye_status == "CLOSED":
                if eye_closed_start is None:
                    eye_closed_start = now

                eye_closed_duration = now - eye_closed_start
            else:
                eye_closed_start = None
                eye_closed_duration = 0.0

            eye_break_warning = (
                eye_closed_duration >= LONG_EYE_CLOSURE_SECONDS
            )

            adaptive_message = get_adaptive_message(
                stable_gaze,
                eye_status,
                eye_closed_duration
            )

            if eye_break_warning:
                message_color = (0, 0, 255)
            elif stable_gaze == "CENTER":
                message_color = (0, 255, 0)
            else:
                message_color = (0, 255, 255)

            # Adaptive response panel
            panel_top = 365
            panel_bottom = 425

            cv2.rectangle(
                frame,
                (10, panel_top),
                (width - 10, panel_bottom),
                (30, 30, 30),
                -1
            )

            cv2.putText(
                frame,
                "ADAPTIVE RESPONSE:",
                (20, 388),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1
            )

            cv2.putText(
                frame,
                adaptive_message[:65],
                (20, 413),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52, message_color, 2
            )

        else:
            # No face detected
            eye_status = "UNKNOWN"
            stable_gesture = "NO FACE"
            stable_gaze = "UNKNOWN"

            closed_frames = 0
            eye_closed_start = None
            eye_closed_duration = 0.0
            eye_break_warning = False

            ear_history.clear()
            gesture_history.clear()
            gaze_history.clear()
            gaze_x_history.clear()
            gaze_y_history.clear()

            adaptive_message = "Face not detected - position yourself in view"

            cv2.putText(
                frame,
                adaptive_message,
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 255), 2
            )

        # ----------------------------------------------
        # DISPLAY
        # ----------------------------------------------

        cv2.putText(
            frame,
            "Head: {}".format(stable_gesture),
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 255), 2
        )

        cv2.putText(
            frame,
            "C: Calibrate | R: Reset | Q: Quit",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2
        )

        cv2.imshow("Neuro-Adaptive Interface", frame)

        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------
        # KEYBOARD CONTROLS
        # ----------------------------------------------

        if key == ord("c"):
            if results.multi_face_landmarks:
                current_landmarks = (
                    results.multi_face_landmarks[0].landmark
                )

                pose = get_head_pose(
                    current_landmarks, width, height
                )

                right_sample = get_eye_gaze(
                    current_landmarks,
                    RIGHT_IRIS_CENTER,
                    RIGHT_EYE_CORNERS,
                    width,
                    height
                )

                left_sample = get_eye_gaze(
                    current_landmarks,
                    LEFT_IRIS_CENTER,
                    LEFT_EYE_CORNERS,
                    width,
                    height
                )

                if pose is not None:
                    calibration_yaw.append(pose[0])
                    calibration_pitch.append(pose[1])

                if (
                    right_sample[0] is not None
                    and right_sample[1] is not None
                    and left_sample[0] is not None
                    and left_sample[1] is not None
                ):
                    calibration_gaze_x.append(
                        (right_sample[0] + left_sample[0]) / 2.0
                    )

                    calibration_gaze_y.append(
                        (right_sample[1] + left_sample[1]) / 2.0
                    )

                print(
                    "Calibration samples - head: {}/{}, gaze: {}/{}".format(
                        len(calibration_yaw),
                        CALIBRATION_SAMPLES,
                        len(calibration_gaze_x),
                        CALIBRATION_SAMPLES
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

            else:
                print(
                    "Face not detected. Keep your face visible and press C again."
                )

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

            print(
                "Calibration reset. Look straight and press C 20 times."
            )

        elif key == ord("q"):
            break

finally:
    camera.release()
    face_mesh.close()
    cv2.destroyAllWindows()
    print("Program closed.")