#Face detection

# import cv2
# import mediapipe as mp
#
# print("MediaPipe version:", mp.__version__)
#
# # Check whether the legacy Solutions API is available
# if not hasattr(mp, "solutions"):
#     print("MediaPipe Solutions API is not available in this version.")
#     print("Please tell me this message if it appears.")
#     raise SystemExit
#
# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils
#
# camera = cv2.VideoCapture(0)
#
# if not camera.isOpened():
#     print("Camera open cheyyan kazhinjilla!")
#     raise SystemExit
#
# try:
#     with mp_face_detection.FaceDetection(
#         model_selection=0,
#         min_detection_confidence=0.5
#     ) as face_detection:
#
#         while True:
#             success, frame = camera.read()
#
#             if not success:
#                 print("Camera frame kittunnilla!")
#                 break
#
#             rgb_frame = cv2.cvtColor(
#                 frame, cv2.COLOR_BGR2RGB
#             )
#
#             results = face_detection.process(rgb_frame)
#
#             if results.detections:
#                 for detection in results.detections:
#                     mp_drawing.draw_detection(frame, detection)
#
#                 text = "Face detected"
#                 color = (0, 255, 0)
#             else:
#                 text = "No face detected"
#                 color = (0, 0, 255)
#
#             cv2.putText(
#                 frame, text, (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1, color, 2
#             )
#
#             cv2.imshow("Neuro-Adaptive Interface", frame)
#
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break
#
# finally:
#     camera.release()
#     cv2.destroyAllWindows()

# Webcam Face Detection + Face Box + Confidence Score.


# import cv2
# import mediapipe as mp
#
# print("MediaPipe version:", mp.__version__)
#
# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils
#
# cap = cv2.VideoCapture(0)
#
# if not cap.isOpened():
#     print("Error: Webcam not found!")
#     exit()
#
# with mp_face_detection.FaceDetection(
#     model_selection=0,
#     min_detection_confidence=0.5
# ) as face_detection:
#
#     while True:
#         success, frame = cap.read()
#
#         if not success:
#             print("Error: Cannot read webcam frame!")
#             break
#
#         # Convert BGR image to RGB
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Detect faces
#         results = face_detection.process(rgb_frame)
#
#         if results.detections:
#             for detection in results.detections:
#                 # Draw face box and keypoints
#                 mp_drawing.draw_detection(frame, detection)
#
#                 # Show confidence score
#                 confidence = detection.score[0]
#                 cv2.putText(
#                     frame,
#                     "Confidence: {:.2f}".format(confidence),
#                     (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.8,
#                     (0, 255, 0),
#                     2
#                 )
#
#         cv2.imshow("NeuroAdaptiveInterface - Face Detection", frame)
#
#         # Press Q to close
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#
# cap.release()
# cv2.destroyAllWindows()
#

# Head Movement Tracking


# import cv2
# import mediapipe as mp
#
# mp_face_mesh = mp.solutions.face_mesh
#
# cap = cv2.VideoCapture(0)
#
# if not cap.isOpened():
#     print("Webcam not found!")
#     exit()
#
# with mp_face_mesh.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# ) as face_mesh:
#
#     while True:
#         success, frame = cap.read()
#
#         if not success:
#             break
#
#         frame = cv2.flip(frame, 1)
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(rgb)
#
#         direction = "Face not detected"
#
#         if results.multi_face_landmarks:
#             face = results.multi_face_landmarks[0]
#             h, w, _ = frame.shape
#
#             # Approximate nose position
#             nose = face.landmark[1]
#             nose_x = int(nose.x * w)
#             nose_y = int(nose.y * h)
#
#             center_x = w // 2
#             center_y = h // 2
#
#             if nose_x < center_x - 35:
#                 direction = "Looking Left"
#             elif nose_x > center_x + 35:
#                 direction = "Looking Right"
#             elif nose_y < center_y - 35:
#                 direction = "Head Up"
#             elif nose_y > center_y + 35:
#                 direction = "Head Down"
#             else:
#                 direction = "Head Center"
#
#             cv2.circle(
#                 frame, (nose_x, nose_y), 6, (0, 255, 0), -1
#             )
#
#         cv2.putText(
#             frame,
#             direction,
#             (20, 40),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             (0, 255, 0),
#             2
#         )
#
#         cv2.imshow("NeuroAdaptiveInterface - Head Tracking", frame)
#
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#
# cap.release()
# cv2.destroyAllWindows()

# Accurate Head Pose Estimation (Yaw, Pitch, Roll)


# import cv2
# import mediapipe as mp
# import numpy as np
#
# mp_face_mesh = mp.solutions.face_mesh
#
# # Approximate 3D facial reference points
# model_points = np.array([
#     (0.0, 0.0, 0.0),           # Nose tip
#     (0.0, -330.0, -65.0),      # Chin
#     (-225.0, 170.0, -135.0),   # Left eye corner
#     (225.0, 170.0, -135.0),    # Right eye corner
#     (-150.0, -150.0, -125.0),  # Left mouth corner
#     (150.0, -150.0, -125.0)    # Right mouth corner
# ], dtype=np.float64)
#
# # MediaPipe landmark indexes
# landmark_ids = [1, 152, 33, 263, 61, 291]
#
# cap = cv2.VideoCapture(0)
#
# if not cap.isOpened():
#     print("Webcam not found!")
#     raise SystemExit
#
# with mp_face_mesh.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# ) as face_mesh:
#
#     while True:
#         success, frame = cap.read()
#
#         if not success:
#             print("Cannot read webcam frame!")
#             break
#
#         frame = cv2.flip(frame, 1)
#         height, width, _ = frame.shape
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(rgb)
#
#         if results.multi_face_landmarks:
#             face = results.multi_face_landmarks[0]
#
#             image_points = np.array([
#                 (
#                     face.landmark[i].x * width,
#                     face.landmark[i].y * height
#                 )
#                 for i in landmark_ids
#             ], dtype=np.float64)
#
#             focal_length = width
#             center = (width / 2, height / 2)
#
#             camera_matrix = np.array([
#                 [focal_length, 0, center[0]],
#                 [0, focal_length, center[1]],
#                 [0, 0, 1]
#             ], dtype=np.float64)
#
#             distortion = np.zeros((4, 1), dtype=np.float64)
#
#             success, rotation_vector, translation_vector = (
#                 cv2.solvePnP(
#                     model_points,
#                     image_points,
#                     camera_matrix,
#                     distortion,
#                     flags=cv2.SOLVEPNP_ITERATIVE
#                 )
#             )
#
#             if success:
#                 rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
#
#                 angles, _, _, _, _, _ = cv2.RQDecomp3x3(
#                     rotation_matrix
#                 )
#
#                 pitch, yaw, roll = angles
#
#                 cv2.putText(
#                     frame,
#                     "Pitch: {:.1f}".format(pitch),
#                     (20, 35),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7, (0, 255, 0), 2
#                 )
#
#                 cv2.putText(
#                     frame,
#                     "Yaw: {:.1f}".format(yaw),
#                     (20, 65),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7, (0, 255, 0), 2
#                 )
#
#                 cv2.putText(
#                     frame,
#                     "Roll: {:.1f}".format(roll),
#                     (20, 95),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7, (0, 255, 0), 2
#                 )
#
#                 # Draw facial reference points
#                 for point in image_points:
#                     cv2.circle(
#                         frame,
#                         (int(point[0]), int(point[1])),
#                         3, (0, 0, 255), -1
#                     )
#
#         cv2.imshow("NeuroAdaptiveInterface - Head Pose", frame)
#
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#
# cap.release()
# cv2.destroyAllWindows()


# Head Gesture Recognition

# from collections import deque, Counter
#
# import cv2
# import mediapipe as mp
# import numpy as np
#
#
# # ---------------------------------
# # 1. MediaPipe Face Mesh
# # ---------------------------------
# mp_face_mesh = mp.solutions.face_mesh
#
# # Approximate 3D facial reference points
# model_points = np.array([
#     (0.0, 0.0, 0.0),            # Nose tip
#     (0.0, -330.0, -65.0),       # Chin
#     (-225.0, 170.0, -135.0),    # Left eye corner
#     (225.0, 170.0, -135.0),     # Right eye corner
#     (-150.0, -150.0, -125.0),   # Left mouth corner
#     (150.0, -150.0, -125.0)     # Right mouth corner
# ], dtype=np.float64)
#
# landmark_ids = [1, 152, 33, 263, 61, 291]
#
#
# # ---------------------------------
# # 2. Webcam Setup
# # ---------------------------------
# cap = cv2.VideoCapture(0)
#
# if not cap.isOpened():
#     print("ERROR: Webcam not found!")
#     raise SystemExit
#
# # Store the last 7 gestures
# gesture_history = deque(maxlen=7)
#
# # Camera matrix is created for each frame's dimensions
# with mp_face_mesh.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# ) as face_mesh:
#
#     while True:
#         success, frame = cap.read()
#
#         if not success:
#             print("ERROR: Cannot read webcam frame!")
#             break
#
#         # Mirror the webcam image
#         frame = cv2.flip(frame, 1)
#
#         height, width, _ = frame.shape
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         results = face_mesh.process(rgb_frame)
#
#         gesture = "FACE NOT DETECTED"
#         stable_gesture = gesture
#
#         # ---------------------------------
#         # 3. Detect Face and Estimate Pose
#         # ---------------------------------
#         if results.multi_face_landmarks:
#             face = results.multi_face_landmarks[0]
#
#             image_points = np.array([
#                 (
#                     face.landmark[i].x * width,
#                     face.landmark[i].y * height
#                 )
#                 for i in landmark_ids
#             ], dtype=np.float64)
#
#             focal_length = width
#             center = (width / 2.0, height / 2.0)
#
#             camera_matrix = np.array([
#                 [focal_length, 0, center[0]],
#                 [0, focal_length, center[1]],
#                 [0, 0, 1]
#             ], dtype=np.float64)
#
#             distortion = np.zeros((4, 1), dtype=np.float64)
#
#             success, rotation_vector, translation_vector = (
#                 cv2.solvePnP(
#                     model_points,
#                     image_points,
#                     camera_matrix,
#                     distortion,
#                     flags=cv2.SOLVEPNP_ITERATIVE
#                 )
#             )
#
#             if success:
#                 rotation_matrix, _ = cv2.Rodrigues(
#                     rotation_vector
#                 )
#
#                 angles, _, _, _, _, _ = cv2.RQDecomp3x3(
#                     rotation_matrix
#                 )
#
#                 pitch, yaw, roll = angles
#
#                 # ---------------------------------
#                 # 4. Head Gesture Recognition
#                 # ---------------------------------
#                 gesture = "CENTER"
#
#                 if yaw < -10:
#                     gesture = "LEFT"
#                 elif yaw > 10:
#                     gesture = "RIGHT"
#                 elif pitch < -10:
#                     gesture = "UP"
#                 elif pitch > 10:
#                     gesture = "DOWN"
#
#                 # ---------------------------------
#                 # 5. Gesture Smoothing
#                 # ---------------------------------
#                 gesture_history.append(gesture)
#
#                 stable_gesture = Counter(
#                     gesture_history
#                 ).most_common(1)[0][0]
#
#                 # ---------------------------------
#                 # 6. Display Angles
#                 # ---------------------------------
#                 cv2.putText(
#                     frame,
#                     "Pitch: {:.1f}".format(pitch),
#                     (20, 35),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7, (0, 255, 0), 2
#                 )
#
#                 cv2.putText(
#                     frame,
#                     "Yaw: {:.1f}".format(yaw),
#                     (20, 65),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7, (0, 255, 0), 2
#                 )
#
#                 cv2.putText(
#                     frame,
#                     "Roll: {:.1f}".format(roll),
#                     (20, 95),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7, (0, 255, 0), 2
#                 )
#
#                 # Draw facial reference points
#                 for point in image_points:
#                     cv2.circle(
#                         frame,
#                         (int(point[0]), int(point[1])),
#                         3, (0, 0, 255), -1
#                     )
#
#         # ---------------------------------
#         # 7. Display Gestures
#         # ---------------------------------
#         cv2.putText(
#             frame,
#             "Gesture: " + gesture,
#             (20, 135),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7, (0, 255, 255), 2
#         )
#
#         cv2.putText(
#             frame,
#             "Stable: " + stable_gesture,
#             (20, 170),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7, (255, 255, 0), 2
#         )
#
#         cv2.imshow(
#             "NeuroAdaptiveInterface - Head Gesture",
#             frame
#         )
#
#         # Press Q to exit
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#
# cap.release()
# cv2.destroyAllWindows()


# Complete Python code



from collections import deque, Counter

import cv2
import mediapipe as mp
import numpy as np


# ==========================================
# NeuroAdaptiveInterface
# Face Detection + Head Pose + Gesture Smoothing
# ==========================================

WINDOW_NAME = "NeuroAdaptiveInterface"

YAW_THRESHOLD = 15.0
PITCH_THRESHOLD = 8.0
SMOOTHING_FRAMES = 5
CALIBRATION_FRAMES = 20

gesture_history = deque(maxlen=SMOOTHING_FRAMES)
calibration_yaw = []
calibration_pitch = []

neutral_yaw = None
neutral_pitch = None

current_yaw = None
current_pitch = None

mp_face_mesh = mp.solutions.face_mesh

# Approximate 3D facial reference points
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye corner
    (225.0, 170.0, -135.0),      # Right eye corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

landmark_ids = [1, 152, 33, 263, 61, 291]


def angle_difference(angle, neutral):
    """Return the shortest signed difference between two angles."""
    return (angle - neutral + 180.0) % 360.0 - 180.0


def detect_gesture(yaw_delta, pitch_delta):
    """Detect head gestures relative to the calibrated neutral position."""

    # CENTER
    if (abs(yaw_delta) <= YAW_THRESHOLD and
            abs(pitch_delta) <= PITCH_THRESHOLD):
        return "CENTER"

    # Left / Right
    if yaw_delta < -YAW_THRESHOLD:
        return "RIGHT"
    elif yaw_delta > YAW_THRESHOLD:
        return "LEFT"

    # UP / DOWN
    # This mapping assumes lowering the head produces a
    # more positive pitch_delta in your camera setup.
    if pitch_delta > PITCH_THRESHOLD:
        return "DOWN"
    elif pitch_delta < -PITCH_THRESHOLD:
        return "UP"

    return "CENTER"


def smooth_gesture(history):
    if not history:
        return "CENTER"
    return Counter(history).most_common(1)[0][0]


def draw_panel(frame, lines):
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (470, 330),
        (25, 25, 25),
        -1
    )

    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

    y = 40
    for text, color in lines:
        cv2.putText(
            frame,
            text,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA
        )
        y += 30


def main():
    global neutral_yaw, neutral_pitch
    global current_yaw, current_pitch

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Webcam not found!")
        return

    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            while True:
                success, frame = cap.read()

                if not success:
                    print("ERROR: Cannot read webcam frame!")
                    break

                frame = cv2.flip(frame, 1)
                height, width, _ = frame.shape

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                face_status = "Face: NOT DETECTED"
                gesture = "NO FACE"
                stable_gesture = "NO FACE"

                pitch = yaw = roll = None
                yaw_delta = pitch_delta = None

                if results.multi_face_landmarks:
                    face = results.multi_face_landmarks[0]

                    image_points = np.array([
                        (
                            face.landmark[i].x * width,
                            face.landmark[i].y * height
                        )
                        for i in landmark_ids
                    ], dtype=np.float64)

                    xs = [
                        int(point.x * width)
                        for point in face.landmark
                    ]
                    ys = [
                        int(point.y * height)
                        for point in face.landmark
                    ]

                    x1 = max(0, min(xs))
                    y1 = max(0, min(ys))
                    x2 = min(width - 1, max(xs))
                    y2 = min(height - 1, max(ys))

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 220, 0),
                        2
                    )

                    for point in image_points:
                        cv2.circle(
                            frame,
                            (int(point[0]), int(point[1])),
                            4,
                            (0, 0, 255),
                            -1
                        )

                    focal_length = float(width)
                    center = (width / 2.0, height / 2.0)

                    camera_matrix = np.array([
                        [focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]
                    ], dtype=np.float64)

                    distortion = np.zeros((4, 1), dtype=np.float64)

                    pose_ok, rotation_vector, translation_vector = (
                        cv2.solvePnP(
                            model_points,
                            image_points,
                            camera_matrix,
                            distortion,
                            flags=cv2.SOLVEPNP_ITERATIVE
                        )
                    )

                    face_status = "Face: DETECTED"

                    if pose_ok:
                        rotation_matrix, _ = cv2.Rodrigues(
                            rotation_vector
                        )

                        angles, _, _, _, _, _ = cv2.RQDecomp3x3(
                            rotation_matrix
                        )

                        pitch, yaw, roll = map(float, angles)

                        current_yaw = yaw
                        current_pitch = pitch

                        if neutral_yaw is None or neutral_pitch is None:
                            gesture = "PRESS C TO CALIBRATE"
                            stable_gesture = gesture
                            gesture_history.clear()

                        else:
                            yaw_delta = angle_difference(
                                current_yaw, neutral_yaw
                            )
                            pitch_delta = angle_difference(
                                current_pitch, neutral_pitch
                            )

                            gesture = detect_gesture(
                                yaw_delta,
                                pitch_delta
                            )

                            gesture_history.append(gesture)
                            stable_gesture = smooth_gesture(
                                gesture_history
                            )

                else:
                    gesture_history.clear()

                panel_lines = [
                    ("NeuroAdaptiveInterface", (0, 255, 255)),
                    (
                        face_status,
                        (0, 255, 0)
                        if results.multi_face_landmarks
                        else (0, 0, 255)
                    )
                ]

                if pitch is not None:
                    panel_lines.extend([
                        ("Yaw:   {:.1f}".format(yaw), (255, 255, 255)),
                        ("Pitch: {:.1f}".format(pitch), (255, 255, 255)),
                        ("Roll:  {:.1f}".format(roll), (255, 255, 255))
                    ])

                    if yaw_delta is not None:
                        panel_lines.extend([
                            (
                                "Yaw Delta: {:.1f}".format(yaw_delta),
                                (255, 255, 0)
                            ),
                            (
                                "Pitch Delta: {:.1f}".format(pitch_delta),
                                (255, 255, 0)
                            )
                        ])

                panel_lines.extend([
                    ("Gesture: " + gesture, (0, 255, 255)),
                    ("Stable: " + stable_gesture, (0, 255, 0)),
                    ("C = Calibrate | Q = Quit", (255, 255, 255))
                ])

                draw_panel(frame, panel_lines)
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF

                # Collect multiple frames for calibration
                if key == ord("c"):
                    if current_yaw is not None and current_pitch is not None:
                        calibration_yaw.append(current_yaw)
                        calibration_pitch.append(current_pitch)

                        if len(calibration_yaw) >= CALIBRATION_FRAMES:
                            neutral_yaw = float(np.mean(calibration_yaw))
                            neutral_pitch = float(np.mean(calibration_pitch))

                            calibration_yaw.clear()
                            calibration_pitch.clear()
                            gesture_history.clear()

                            print(
                                "Calibrated: yaw={:.1f}, pitch={:.1f}".format(
                                    neutral_yaw, neutral_pitch
                                )
                            )
                        else:
                            print(
                                "Calibration samples: {}/{}".format(
                                    len(calibration_yaw),
                                    CALIBRATION_FRAMES
                                )
                            )

                if key == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
