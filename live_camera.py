
import threading
import time

import av
import cv2
import mediapipe as mp
import streamlit as st

from streamlit_webrtc import webrtc_streamer, WebRtcMode
from streamlit_autorefresh import st_autorefresh


# --------------------------------------------------
# SHARED LIVE CAMERA STATE
# --------------------------------------------------

_state_lock = threading.Lock()

_live_state = {
    "face_detected": False,
    "blink_count": 0,
    "gaze_state": "Waiting for camera",
    "head_direction": "Waiting for camera",
    "last_blink_time": 0.0,
    "eye_closed_frames": 0,
    "frames": 0,
    "look_away_count": 0,
    "start_time": None,
}

_thread_local = threading.local()

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


# --------------------------------------------------
# MEDIAPIPE FACE MESH
# --------------------------------------------------

def get_face_mesh():
    if not hasattr(_thread_local, "face_mesh"):
        _thread_local.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )

    return _thread_local.face_mesh


# --------------------------------------------------
# EYE ASPECT RATIO
# --------------------------------------------------

def distance(point_a, point_b):
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return (dx * dx + dy * dy) ** 0.5


def eye_aspect_ratio(landmarks, indices, width, height):
    points = []

    for index in indices:
        landmark = landmarks[index]
        points.append((
            int(landmark.x * width),
            int(landmark.y * height),
        ))

    vertical_1 = distance(points[1], points[5])
    vertical_2 = distance(points[2], points[4])
    horizontal = distance(points[0], points[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


# --------------------------------------------------
# APPROXIMATE HEAD DIRECTION
# --------------------------------------------------

def estimate_head_direction(landmarks):
    nose_x = landmarks[1].x
    left_eye_x = landmarks[33].x
    right_eye_x = landmarks[263].x

    eye_midpoint = (left_eye_x + right_eye_x) / 2.0
    eye_width = abs(right_eye_x - left_eye_x)

    if eye_width < 0.001:
        return "Head direction unavailable"

    offset = (nose_x - eye_midpoint) / eye_width

    if offset > 0.08:
        return "Head turned Right"

    if offset < -0.08:
        return "Head turned Left"

    return "Head Center"


# --------------------------------------------------
# PROCESS WEBCAM FRAME
# --------------------------------------------------

def process_frame(frame):
    image = frame.to_ndarray(format="bgr24")

    # Mirror the webcam video horizontally
    image = cv2.flip(image, 1)

    height, width = image.shape[:2]

    if width > 640:
        new_height = int(height * 640 / width)
        image = cv2.resize(image, (640, new_height))
        height, width = image.shape[:2]

    # Improve visibility in dark webcam frames.
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
    image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = get_face_mesh().process(rgb_image)

    now = time.time()

    with _state_lock:
        _live_state["frames"] += 1

        if _live_state["start_time"] is None:
            _live_state["start_time"] = now

        if not results.multi_face_landmarks:
            _live_state["face_detected"] = False
            _live_state["gaze_state"] = "Face not detected"
            _live_state["head_direction"] = "Face not detected"

            cv2.putText(
                image, "Face not detected", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255), 2,
            )

        else:
            landmarks = results.multi_face_landmarks[0].landmark
            _live_state["face_detected"] = True

            # Blink detection
            left_ear = eye_aspect_ratio(
                landmarks, LEFT_EYE, width, height
            )
            right_ear = eye_aspect_ratio(
                landmarks, RIGHT_EYE, width, height
            )
            average_ear = (left_ear + right_ear) / 2.0

            if average_ear < 0.20:
                _live_state["eye_closed_frames"] += 1
            else:
                closed_frames = _live_state["eye_closed_frames"]

                if (
                    closed_frames >= 1
                    and now - _live_state["last_blink_time"] > 0.20
                ):
                    _live_state["blink_count"] += 1
                    _live_state["last_blink_time"] = now

                _live_state["eye_closed_frames"] = 0

            # Approximate gaze estimation
            try:
                left_iris = landmarks[468]
                right_iris = landmarks[473]

                left_ratio = (
                    left_iris.x - landmarks[33].x
                ) / max(
                    abs(landmarks[133].x - landmarks[33].x),
                    0.001,
                )

                right_ratio = (
                    right_iris.x - landmarks[362].x
                ) / max(
                    abs(landmarks[263].x - landmarks[362].x),
                    0.001,
                )

                iris_ratio = (left_ratio + right_ratio) / 2.0

                if 0.35 <= iris_ratio <= 0.65:
                    gaze = "Approximate center gaze"
                else:
                    gaze = "Possible look-away"

                if (
                    gaze == "Possible look-away"
                    and _live_state["gaze_state"] != gaze
                ):
                    _live_state["look_away_count"] += 1

                _live_state["gaze_state"] = gaze

            except (IndexError, AttributeError):
                _live_state["gaze_state"] = "Gaze unavailable"

            # Head direction estimation
            head_direction = estimate_head_direction(landmarks)
            _live_state["head_direction"] = head_direction

            # Draw selected landmarks
            for landmark in landmarks[::8]:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(
                    image, (x, y), 2, (56, 189, 248), -1
                )

            cv2.putText(
                image, "Face detected", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0), 2,
            )

            cv2.putText(
                image, "EAR: {:.2f}".format(average_ear), (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2,
            )

            cv2.putText(
                image, _live_state["gaze_state"], (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 220, 255), 2,
            )

            cv2.putText(
                image, head_direction, (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 200, 0), 2,
            )

    return av.VideoFrame.from_ndarray(image, format="bgr24")


# --------------------------------------------------
# STREAMLIT CAMERA UI
# --------------------------------------------------

def render_live_camera():
    # Refresh Streamlit UI once per second.
    st_autorefresh(
        interval=1000,
        key="neuroadaptive_camera_refresh",
    )

    st.divider()
    st.header("📷 Live Webcam Monitoring")

    st.caption(
        "Experimental blink, approximate gaze and head-direction "
        "measurements. These signals do not diagnose cognitive "
        "overload or any medical condition."
    )

    st.warning(
        "Camera frames are sent to the application server for "
        "processing. Start the camera only if you consent."
    )

    rtc_config = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }

    webrtc_streamer(
        key="neuroadaptive-live-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_frame_callback=process_frame,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    with _state_lock:
        face_detected = _live_state["face_detected"]
        blink_count = _live_state["blink_count"]
        look_away_count = _live_state["look_away_count"]
        gaze_state = _live_state["gaze_state"]
        head_direction = _live_state["head_direction"]
        start_time = _live_state["start_time"]

    duration = (
        0.0
        if start_time is None
        else max(0.0, time.time() - start_time)
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Face status",
        "Detected" if face_detected else "Not detected",
    )
    c2.metric("Blink count", blink_count)
    c3.metric("Possible look-away events", look_away_count)
    c4.metric("Monitoring time (sec)", "{:.1f}".format(duration))

    st.info("Current gaze estimate: " + gaze_state)
    st.info("Current head direction: " + head_direction)
