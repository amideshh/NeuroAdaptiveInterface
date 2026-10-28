import os
import time
import threading
from datetime import datetime
from io import StringIO

import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import requests
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    VideoProcessorBase
)

from streamlit_autorefresh import st_autorefresh


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NeuroAdaptiveInterface",
    page_icon="🧠",
    layout="wide"
)

st_autorefresh(
    interval=1000,
    limit=None,
    key="face_demo_refresh"
)


# =========================================================
# CONFIGURATION
# =========================================================

MIN_CALIBRATION_SAMPLES = 20
LOOK_AWAY_THRESHOLD = 0.15
EAR_THRESHOLD = 0.21
LONG_EYE_CLOSED_SECONDS = 2.0

mp_face_mesh = mp.solutions.face_mesh


# =========================================================
# GITHUB CSV STORAGE
# =========================================================

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH", "main")

EVENT_FILE = "session_logs/event_log.csv"
SUMMARY_FILE = "session_logs/session_summary.csv"


def github_headers():
    return {
        "Authorization": "Bearer " + GITHUB_TOKEN,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }


def github_enabled():
    return bool(GITHUB_TOKEN and GITHUB_REPO)


def append_csv(filename, rows):
    """
    Append rows to a CSV stored in GitHub.
    """
    if not rows or not github_enabled():
        return False

    url = (
        "https://api.github.com/repos/"
        + GITHUB_REPO
        + "/contents/"
        + filename
    )

    try:
        response = requests.get(
            url,
            headers=github_headers(),
            params={"ref": GITHUB_BRANCH},
            timeout=15
        )

        sha = None
        old_content = ""

        if response.status_code == 200:
            data = response.json()
            sha = data.get("sha")
            import base64

            old_content = base64.b64decode(
                data["content"]
            ).decode("utf-8")

        elif response.status_code != 404:
            return False

        old_df = pd.DataFrame()

        if old_content.strip():
            old_df = pd.read_csv(StringIO(old_content))

        new_df = pd.DataFrame(rows)

        combined = pd.concat(
            [old_df, new_df],
            ignore_index=True
        )

        content = combined.to_csv(index=False)

        import base64

        payload = {
            "message": "Update " + filename,
            "content": base64.b64encode(
                content.encode("utf-8")
            ).decode("utf-8"),
            "branch": GITHUB_BRANCH
        }

        if sha:
            payload["sha"] = sha

        result = requests.put(
            url,
            headers=github_headers(),
            json=payload,
            timeout=20
        )

        return result.status_code in (200, 201)

    except Exception:
        return False


# =========================================================
# FACE MEASUREMENT FUNCTIONS
# =========================================================

def calculate_ear(landmarks, indices):
    """
    Eye Aspect Ratio.
    """
    try:
        points = np.array([
            [landmarks[i].x, landmarks[i].y]
            for i in indices
        ])

        vertical_1 = np.linalg.norm(
            points[1] - points[5]
        )

        vertical_2 = np.linalg.norm(
            points[2] - points[4]
        )

        horizontal = np.linalg.norm(
            points[0] - points[3]
        )

        if horizontal < 1e-6:
            return None

        return float(
            (vertical_1 + vertical_2)
            / (2.0 * horizontal)
        )

    except (IndexError, ValueError):
        return None


def get_eye_ear(landmarks):
    left_eye = [362, 385, 387, 263, 373, 380]
    right_eye = [33, 160, 158, 133, 153, 144]

    left = calculate_ear(landmarks, left_eye)
    right = calculate_ear(landmarks, right_eye)

    values = [
        value for value in (left, right)
        if value is not None
    ]

    if not values:
        return None

    return float(np.mean(values))


def head_pose(landmarks):
    """
    Approximate yaw and pitch from facial landmark geometry.
    These are relative tracking signals, not clinical measurements.
    """
    try:
        nose = landmarks[1]
        left_face = landmarks[234]
        right_face = landmarks[454]
        forehead = landmarks[10]
        chin = landmarks[152]

        face_width = abs(
            right_face.x - left_face.x
        )

        face_height = abs(
            chin.y - forehead.y
        )

        if face_width < 1e-6 or face_height < 1e-6:
            return None, None

        center_x = (
            left_face.x + right_face.x
        ) / 2.0

        center_y = (
            forehead.y + chin.y
        ) / 2.0

        yaw = (
            (nose.x - center_x) / face_width
        ) * 90.0

        pitch = (
            (nose.y - center_y) / face_height
        ) * 90.0

        return float(yaw), float(pitch)

    except (IndexError, ValueError):
        return None, None


def gaze_values(landmarks):
    """
    Estimate relative iris position using refined FaceMesh landmarks.
    """
    try:
        if len(landmarks) < 478:
            return None, None

        def one(iris, corner1, corner2, top, bottom):
            x1 = landmarks[corner1].x
            x2 = landmarks[corner2].x

            y1 = landmarks[top].y
            y2 = landmarks[bottom].y

            eye_width = abs(x2 - x1)
            eye_height = abs(y2 - y1)

            if eye_width < 1e-6 or eye_height < 1e-6:
                return None, None

            gx = (
                landmarks[iris].x - min(x1, x2)
            ) / eye_width

            gy = (
                landmarks[iris].y - min(y1, y2)
            ) / eye_height

            return float(gx), float(gy)

        left = one(468, 362, 263, 386, 374)
        right = one(473, 33, 133, 159, 145)

        valid = [
            value for value in (left, right)
            if value[0] is not None
            and value[1] is not None
        ]

        if not valid:
            return None, None

        return (
            float(np.mean([v[0] for v in valid])),
            float(np.mean([v[1] for v in valid]))
        )

    except (IndexError, ValueError):
        return None, None


# =========================================================
# FACE PROCESSOR
# =========================================================

class FaceProcessor(VideoProcessorBase):

    def __init__(self):
        self.lock = threading.RLock()

        self.session_id = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        self.start_time = time.time()
        self.last_frame_time = time.time()

        self.face_detected = False
        self.last_yaw = None
        self.last_pitch = None
        self.last_ear = None

        self.calibrated = False
        self.calibrate_requested = False
        self.calibration_status = "Not started"

        self.cyaw = []
        self.cpitch = []
        self.cgx = []
        self.cgy = []

        self.nyaw = 0.0
        self.npitch = 0.0
        self.ngx = 0.5
        self.ngy = 0.5

        self.direction = "Uncalibrated"
        self.gaze = "Unknown"

        self.blink_count = 0
        self.lookaway_count = 0

        self.eye_closed = False
        self.eye_closed_start = None
        self.long_eye_warning = False

        self.lookaway_seconds = 0.0
        self.center_seconds = 0.0
        self.unknown_seconds = 0.0

        self.last_direction = "Unknown"
        self.direction_counts = {
            "Center": 0,
            "Left": 0,
            "Right": 0,
            "Up": 0,
            "Down": 0
        }

        self.events = []
        self.last_event = None

        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def log(self, event, details=""):
        now = datetime.now()

        self.events.append({
            "Session_ID": self.session_id,
            "Timestamp": now.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Elapsed_seconds": round(
                time.time() - self.start_time, 2
            ),
            "Event": event,
            "Details": str(details)
        })

        self.last_event = event

    def request_calibration(self):
        with self.lock:
            self.calibrated = False
            self.calibrate_requested = True
            self.calibration_status = "Collecting samples..."

            self.cyaw = []
            self.cpitch = []
            self.cgx = []
            self.cgy = []

            self.log(
                "CALIBRATION_STARTED",
                "Look straight at the camera and hold still."
            )

    def reset_calibration(self):
        with self.lock:
            self.calibrated = False
            self.calibrate_requested = False
            self.calibration_status = "Not started"

            self.cyaw = []
            self.cpitch = []
            self.cgx = []
            self.cgy = []

            self.nyaw = 0.0
            self.npitch = 0.0
            self.ngx = 0.5
            self.ngy = 0.5

            self.direction = "Uncalibrated"
            self.gaze = "Unknown"

            self.log("CALIBRATION_RESET")

    def snapshot(self):
        with self.lock:
            duration = time.time() - self.start_time

            return {
                "Session_ID": self.session_id,
                "Duration_seconds": round(duration, 2),
                "Face_Detected": self.face_detected,
                "Direction_Counts": dict(self.direction_counts),
                "Yaw": self.last_yaw,
                "Pitch": self.last_pitch,
                "EAR": self.last_ear,
                "Calibrated": self.calibrated,
                "Calibration_Status": self.calibration_status,
                "Head_Samples": len(self.cyaw),
                "Gaze_Samples": len(self.cgx),
                "Direction": self.direction,
                "Gaze": self.gaze,
                "Blink_Count": self.blink_count,
                "Look_Away_Count": self.lookaway_count,
                "Center_Seconds": round(self.center_seconds, 2),
                "Look_Away_Seconds": round(
                    self.lookaway_seconds, 2
                ),
                "Unknown_Gaze_Seconds": round(
                    self.unknown_seconds, 2
                ),
                "Last_Event": self.last_event
            }

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Flip preview horizontally; no facial drawing or text overlay.
        image = cv2.flip(image, 1)

        rgb = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )

        result = self.face_mesh.process(rgb)

        now = time.time()
        dt = min(max(now - self.last_frame_time, 0), 0.5)
        self.last_frame_time = now

        with self.lock:
            self.face_detected = bool(
                result.multi_face_landmarks
            )

            if not result.multi_face_landmarks:
                self.unknown_seconds += dt
                self.direction = "Unknown"
                self.gaze = "Unknown"

                if self.eye_closed:
                    self.eye_closed = False
                    self.eye_closed_start = None

                return av.VideoFrame.from_ndarray(
                    image, format="bgr24"
                )

            landmarks = (
                result.multi_face_landmarks[0].landmark
            )

            yaw, pitch = head_pose(landmarks)
            ear = get_eye_ear(landmarks)
            gx, gy = gaze_values(landmarks)

            self.last_yaw = yaw
            self.last_pitch = pitch
            self.last_ear = ear

            # Calibration: collect only valid samples.
            if self.calibrate_requested:
                if yaw is not None and pitch is not None:
                    self.cyaw.append(yaw)
                    self.cpitch.append(pitch)

                if gx is not None and gy is not None:
                    self.cgx.append(gx)
                    self.cgy.append(gy)

                head_n = len(self.cyaw)
                gaze_n = len(self.cgx)

                self.calibration_status = (
                    "Head samples: {}/{} | Gaze samples: {}/{}"
                    .format(
                        min(head_n, MIN_CALIBRATION_SAMPLES),
                        MIN_CALIBRATION_SAMPLES,
                        min(gaze_n, MIN_CALIBRATION_SAMPLES),
                        MIN_CALIBRATION_SAMPLES
                    )
                )

                if (
                    head_n >= MIN_CALIBRATION_SAMPLES
                    and gaze_n >= MIN_CALIBRATION_SAMPLES
                ):
                    self.nyaw = float(np.median(self.cyaw))
                    self.npitch = float(np.median(self.cpitch))
                    self.ngx = float(np.median(self.cgx))
                    self.ngy = float(np.median(self.cgy))

                    self.calibrated = True
                    self.calibrate_requested = False
                    self.calibration_status = "Calibration complete"

                    self.log(
                        "CALIBRATION_COMPLETED",
                        "Head and gaze baselines saved."
                    )

            # Head direction tracking.
            if self.calibrated and yaw is not None and pitch is not None:
                dyaw = yaw - self.nyaw
                dpitch = pitch - self.npitch

                if dyaw < -10:
                    self.direction = "Left"
                elif dyaw > 10:
                    self.direction = "Right"
                elif dpitch < -8:
                    self.direction = "Up"
                elif dpitch > 8:
                    self.direction = "Down"
                else:
                    self.direction = "Center"

                self.direction_counts[self.direction] = (
                    self.direction_counts.get(
                        self.direction, 0
                    ) + 1
                )

                if self.direction == "Center":
                    self.center_seconds += dt
                else:
                    self.lookaway_seconds += dt

                    if self.last_direction == "Center":
                        self.lookaway_count += 1
                        self.log(
                            "LOOK_AWAY_WARNING",
                            "Head direction: " + self.direction
                        )

                self.last_direction = self.direction

            else:
                self.direction = "Uncalibrated"
                self.unknown_seconds += dt

            # Gaze tracking.
            if self.calibrated and gx is not None and gy is not None:
                dx = gx - self.ngx
                dy = gy - self.ngy

                if abs(dx) > LOOK_AWAY_THRESHOLD:
                    self.gaze = "Away"
                elif abs(dy) > LOOK_AWAY_THRESHOLD:
                    self.gaze = "Away"
                else:
                    self.gaze = "Center"
            else:
                self.gaze = "Unknown"

            # Blink and prolonged eye closure tracking.
            if ear is not None:
                if ear < EAR_THRESHOLD:
                    if not self.eye_closed:
                        self.eye_closed = True
                        self.eye_closed_start = now
                else:
                    if self.eye_closed:
                        self.blink_count += 1

                        closed_duration = (
                            now - self.eye_closed_start
                            if self.eye_closed_start
                            else 0
                        )

                        if (
                            closed_duration
                            >= LONG_EYE_CLOSED_SECONDS
                        ):
                            self.log(
                                "EYE_CLOSURE_WARNING",
                                "Eyes closed for {:.2f} seconds".format(
                                    closed_duration
                                )
                            )

                        else:
                            self.log("BLINK")

                        self.eye_closed = False
                        self.eye_closed_start = None

        # Return clean camera image without landmarks or text.
        return av.VideoFrame.from_ndarray(
            image, format="bgr24"
        )


# =========================================================
# SAVE SESSION
# =========================================================

def save_session(proc):
    data = proc.snapshot()

    summary = {
        "Session_ID": data["Session_ID"],
        "duration_seconds": data["Duration_seconds"],
        "Face_Detected": data["Face_Detected"],
        "Calibrated": data["Calibrated"],
        "Look_Away_Count": data["Look_Away_Count"],
        "Blink_Count": data["Blink_Count"],
        "Look_Away_Seconds": data["Look_Away_Seconds"],
        "Center_Count": proc.direction_counts["Center"],
        "Left_Count": proc.direction_counts["Left"],
        "Right_Count": proc.direction_counts["Right"],
        "Up_Count": proc.direction_counts["Up"],
        "Down_Count": proc.direction_counts["Down"],
        "Center_Seconds": data["Center_Seconds"],
        "Unknown_Gaze_Seconds": data["Unknown_Gaze_Seconds"],
        "Last_Event": data["Last_Event"]
    }

    event_rows = list(proc.events)

    local_dir = "session_logs"
    os.makedirs(local_dir, exist_ok=True)

    summary_path = os.path.join(
        local_dir, "session_summary.csv"
    )

    event_path = os.path.join(
        local_dir, "event_log.csv"
    )

    summary_df = pd.DataFrame([summary])

    if os.path.exists(summary_path):
        old_df = pd.read_csv(summary_path)
        old_df = old_df[
            old_df["Session_ID"] != data["Session_ID"]
        ]
        summary_df = pd.concat(
            [old_df, summary_df],
            ignore_index=True
        )

    summary_df.to_csv(summary_path, index=False)

    if event_rows:
        events_df = pd.DataFrame(event_rows)

        if os.path.exists(event_path):
            old_events = pd.read_csv(event_path)
            events_df = pd.concat(
                [old_events, events_df],
                ignore_index=True
            )

        events_df.to_csv(event_path, index=False)

    github_summary = append_csv(
        SUMMARY_FILE, [summary]
    )

    github_events = append_csv(
        EVENT_FILE, event_rows
    )

    return github_summary, github_events


# =========================================================
# STREAMLIT USER INTERFACE
# =========================================================

st.title("🧠 NeuroAdaptiveInterface")

st.caption(
    "Camera-based head direction, gaze and eye-closure signal demo. "
    "These signals are experimental and do not diagnose cognitive overload."
)

st.info(
    "Allow camera access. Sit in front of the camera, "
    "look straight at it, and click Calibrate."
)

ctx = webrtc_streamer(
    key="neuroadaptive-face-demo",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=FaceProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)

proc = ctx.video_processor

if proc is not None:

    st.divider()
    st.subheader("Calibration")

    col1, col2 = st.columns(2)

    if col1.button(
        "🎯 Calibrate / Recalibrate",
        use_container_width=True
    ):
        proc.request_calibration()

    if col2.button(
        "↩️ Reset calibration",
        use_container_width=True
    ):
        proc.reset_calibration()

    data = proc.snapshot()

    st.caption(
        "Calibration status: "
        + data["Calibration_Status"]
    )

    st.progress(
        min(
            data["Head_Samples"] / MIN_CALIBRATION_SAMPLES,
            1.0
        )
    )

    st.caption(
        "Head samples: {} / {} | Gaze samples: {} / {}".format(
            data["Head_Samples"],
            MIN_CALIBRATION_SAMPLES,
            data["Gaze_Samples"],
            MIN_CALIBRATION_SAMPLES
        )
    )

    if data["Calibrated"]:
        st.success("Calibration completed successfully.")
    elif ctx.state.playing:
        st.warning(
            "If sample counts do not increase, check that the face "
            "is visible and that the camera is sending frames."
        )

    st.divider()
    st.subheader("Live Session Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Face detected",
        "Yes" if data["Face_Detected"] else "No"
    )

    c2.metric(
        "Calibration",
        "Ready" if data["Calibrated"] else "Not ready"
    )

    c3.metric(
        "Session duration",
        "{} sec".format(data["Duration_seconds"])
    )

    c4, c5, c6 = st.columns(3)

    c4.metric(
        "Head direction",
        data["Direction"]
    )

    c5.metric(
        "Gaze signal",
        data["Gaze"]
    )

    c6.metric(
        "Blink count",
        data["Blink_Count"]
    )

    c7, c8, c9 = st.columns(3)

    c7.metric(
        "Look-away count",
        data["Look_Away_Count"]
    )

    c8.metric(
        "Yaw",
        "N/A" if data["Yaw"] is None
        else round(data["Yaw"], 2)
    )

    c9.metric(
        "Pitch",
        "N/A" if data["Pitch"] is None
        else round(data["Pitch"], 2)
    )

    st.divider()
    st.subheader("Session Signal Summary")

    signal_df = pd.DataFrame([
        {
            "Signal": "Center time",
            "Seconds": data["Center_Seconds"]
        },
        {
            "Signal": "Look-away time",
            "Seconds": data["Look_Away_Seconds"]
        },
        {
            "Signal": "Unknown time",
            "Seconds": data["Unknown_Gaze_Seconds"]
        }
    ])

    st.dataframe(
        signal_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Session ID: " + data["Session_ID"]
    )

    if st.button(
        "💾 Save Session",
        use_container_width=True
    ):
        github_summary, github_events = save_session(proc)

        st.success(
            "Session saved locally. "
            "GitHub summary upload: {}. "
            "GitHub event upload: {}.".format(
                "Success" if github_summary else "Not uploaded",
                "Success" if github_events else "Not uploaded"
            )
        )

    with st.expander("Recent session events"):
        if proc.events:
            st.dataframe(
                pd.DataFrame(proc.events).tail(20),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No events recorded yet.")

else:
    st.warning(
        "Start the camera above. Once the video processor starts, "
        "the calibration and metrics will appear here."
    )

st.divider()

st.caption(
    "Privacy: video is processed in the active app session. "
    "This code does not intentionally upload video frames to GitHub; "
    "only CSV session summaries and events are saved."
)