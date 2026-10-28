import os
import time
import base64
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
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode

st.set_page_config(page_title="NeuroAdaptiveInterface - Face Demo", page_icon="👁️", layout="wide")

LOG_DIR = "session_logs"
EVENT_PATH = f"{LOG_DIR}/event_log.csv"
SUMMARY_PATH = f"{LOG_DIR}/session_summary.csv"
EVENT_COLUMNS = ["Session_ID", "Timestamp", "Elapsed_seconds", "Event", "Details"]
SUMMARY_COLUMNS = [
    "Session_ID", "Start_Time", "End_Time", "Duration_seconds", "Blink_Count",
    "Look_Away_Count", "Eye_Closure_Warning_Count", "Center_Gaze_Seconds",
    "Look_Away_Seconds", "Unknown_Gaze_Seconds", "Eyes_Closed_Seconds",
    "Data_Quality", "Notes"
]

EAR_CLOSED_THRESHOLD = 0.20
EYE_CLOSED_WARNING_SECONDS = 2.0
LOOK_AWAY_WARNING_SECONDS = 2.0
HEAD_YAW_THRESHOLD = 18.0
HEAD_PITCH_THRESHOLD = 15.0
GAZE_THRESHOLD = 0.045
CALIBRATION_HEAD_SAMPLES = 20
CALIBRATION_GAZE_SAMPLES = 20
MAX_LOG_ROWS = 10000
mp_face_mesh = mp.solutions.face_mesh


def github_config():
    try:
        token = st.secrets.get("GITHUB_TOKEN", "")
        repo = st.secrets.get("GITHUB_REPO", "")
        branch = st.secrets.get("GITHUB_BRANCH", "main")
    except Exception:
        token, repo, branch = os.getenv("GITHUB_TOKEN", ""), os.getenv("GITHUB_REPO", ""), os.getenv("GITHUB_BRANCH", "main")
    return (token or os.getenv("GITHUB_TOKEN", "")).strip(), (repo or os.getenv("GITHUB_REPO", "")).strip(), (branch or os.getenv("GITHUB_BRANCH", "main")).strip() or "main"


def github_headers(token):
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}


def github_read(path):
    token, repo, branch = github_config()
    if not token or not repo:
        return "", None
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    try:
        response = requests.get(url, headers=github_headers(token), params={"ref": branch}, timeout=15)
        if response.status_code == 404:
            return "", None
        response.raise_for_status()
        payload = response.json()
        content = payload.get("content", "")
        return (base64.b64decode(content).decode("utf-8") if content else ""), payload.get("sha")
    except Exception as exc:
        st.warning(f"Could not read {path} from GitHub: {exc}")
        return "", None


def github_write(path, content):
    token, repo, branch = github_config()
    if not token or not repo:
        return False, "GitHub Secrets are not configured."
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    try:
        current = requests.get(url, headers=github_headers(token), params={"ref": branch}, timeout=15)
        sha = current.json().get("sha") if current.status_code == 200 else None
        if current.status_code not in (200, 404):
            current.raise_for_status()
        body = {
            "message": f"Update {path} from NeuroAdaptiveInterface",
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        if sha:
            body["sha"] = sha
        response = requests.put(url, headers=github_headers(token), json=body, timeout=20)
        if response.status_code not in (200, 201):
            return False, f"GitHub API returned {response.status_code}: {response.text[:300]}"
        return True, "Saved to GitHub."
    except Exception as exc:
        return False, str(exc)


def load_csv(path, columns):
    text, _ = github_read(path)
    try:
        frame = pd.read_csv(StringIO(text)) if text.strip() else (pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=columns))
    except Exception:
        frame = pd.DataFrame(columns=columns)
    for col in columns:
        if col not in frame.columns:
            frame[col] = ""
    return frame[columns]


def append_csv(path, rows, columns):
    if not rows:
        return True, "No rows to save."
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    combined = pd.concat([load_csv(path, columns), pd.DataFrame(rows)], ignore_index=True)
    for col in columns:
        if col not in combined.columns:
            combined[col] = ""
    combined = combined[columns]
    if path == SUMMARY_PATH:
        combined = combined.drop_duplicates(subset=["Session_ID"], keep="last")
    if len(combined) > MAX_LOG_ROWS:
        combined = combined.tail(MAX_LOG_ROWS)
    csv_text = combined.to_csv(index=False)
    combined.to_csv(path, index=False)
    token, repo, _ = github_config()
    if token and repo:
        return github_write(path, csv_text)
    return True, "Saved locally. Configure GITHUB_TOKEN and GITHUB_REPO to persist on GitHub."


def now_string():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def xy(lm, i, w, h):
    return np.array([lm[i].x * w, lm[i].y * h], dtype=np.float32)


def eye_aspect_ratio(lm, indices, w, h):
    p = [xy(lm, i, w, h) for i in indices]
    horizontal = np.linalg.norm(p[0] - p[3])
    if horizontal <= 1e-6:
        return 0.0
    return float((np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2.0 * horizontal))


LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def head_pose(lm, w, h):
    image_points = np.array([
        (lm[1].x*w, lm[1].y*h), (lm[152].x*w, lm[152].y*h),
        (lm[33].x*w, lm[33].y*h), (lm[263].x*w, lm[263].y*h),
        (lm[61].x*w, lm[61].y*h), (lm[291].x*w, lm[291].y*h)
    ], dtype=np.float64)
    model_points = np.array([
        (0, 0, 0), (0, -63.6, -12.5), (-43.3, 32.7, -26),
        (43.3, 32.7, -26), (-28.9, -28.9, -24.1), (28.9, -28.9, -24.1)
    ], dtype=np.float64)
    focal = float(w)
    camera = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float64)
    try:
        ok, rvec, _ = cv2.solvePnP(model_points, image_points, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None, None
        angles = cv2.RQDecomp3x3(cv2.Rodrigues(rvec)[0])[0]
        return float(angles[1]), float(angles[0])  # yaw, pitch
    except Exception:
        return None, None


def gaze_values(lm):
    if len(lm) < 478:
        return None, None

    def one(iris, corner1, corner2, top, bottom):
        x1, x2 = lm[corner1].x, lm[corner2].x
        y1, y2 = lm[top].y, lm[bottom].y
        ew, eh = abs(x2-x1), abs(y2-y1)
        if ew < 1e-6 or eh < 1e-6:
            return None, None
        return (float((lm[iris].x-min(x1, x2))/ew), float((lm[iris].y-min(y1, y2))/eh))
    a = one(468, 362, 263, 386, 374)
    b = one(473, 33, 133, 159, 145)
    xs = [v[0] for v in (a, b) if v[0] is not None]
    ys = [v[1] for v in (a, b) if v[1] is not None]
    return (float(np.mean(xs)), float(np.mean(ys))) if xs and ys else (None, None)


class FaceProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.RLock()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_epoch = time.time()
        self.start_time = now_string()
        self.last_frame_time = None
        self.last_log_time = 0
        self.face_detected = False
        self.calibrated = False
        self.calibrate_requested = False
        self.cyaw, self.cpitch, self.cgx, self.cgy = [], [], [], []
        self.nyaw, self.npitch, self.ngx, self.ngy = 0.0, 0.0, 0.5, 0.5
        self.blinks = self.lookaways = self.eye_warnings = 0
        self.center_seconds = self.lookaway_seconds = self.unknown_seconds = self.closed_seconds = 0.0
        self.eye_closed_since = self.lookaway_since = None
        self.was_eye_closed = False
        self.was_lookaway = False
        self.last_direction, self.last_gaze = "Unknown", "Unknown"
        self.last_yaw = self.last_pitch = self.last_ear = None
        self.direction_counts = {"Center": 0, "Left": 0, "Right": 0, "Up": 0, "Down": 0}
        self.notes, self.events = [], []
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def log(self, event, details=""):
        self.events.append({
            "Session_ID": self.session_id, "Timestamp": now_string(),
            "Elapsed_seconds": round(time.time()-self.start_epoch, 2),
            "Event": event, "Details": str(details)
        })

    def request_calibration(self):
        with self.lock:
            self.calibrated = False
            self.calibrate_requested = True
            self.cyaw, self.cpitch, self.cgx, self.cgy = [], [], [], []
            self.log("CALIBRATION_STARTED", "Look straight at camera and hold still.")

    def reset_calibration(self):
        with self.lock:
            self.calibrated = self.calibrate_requested = False
            self.cyaw, self.cpitch, self.cgx, self.cgy = [], [], [], []
            self.nyaw, self.npitch, self.ngx, self.ngy = 0.0, 0.0, 0.5, 0.5
            self.log("CALIBRATION_RESET", "Calibration baseline cleared.")

    def snapshot(self):
        with self.lock:
            duration = max(0, time.time()-self.start_epoch)
            return {
                "Session_ID": self.session_id, "Start_Time": self.start_time,
                "End_Time": now_string(), "Duration_seconds": round(duration, 2),
                "Blink_Count": self.blinks, "Look_Away_Count": self.lookaways,
                "Eye_Closure_Warning_Count": self.eye_warnings,
                "Center_Gaze_Seconds": round(self.center_seconds, 2),
                "Look_Away_Seconds": round(self.lookaway_seconds, 2),
                "Unknown_Gaze_Seconds": round(max(0, duration-self.center_seconds-self.lookaway_seconds), 2),
                "Eyes_Closed_Seconds": round(self.closed_seconds, 2),
                "Data_Quality": "Calibrated" if self.calibrated else "Not calibrated",
                "Notes": "; ".join(self.notes) if self.notes else "No configured warnings"
            }

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        h, w = image.shape[:2]
        now = time.time()
        dt = 0 if self.last_frame_time is None else min(now-self.last_frame_time, 0.25)
        self.last_frame_time = now
        try:
            result = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        except Exception:
            result = None

        with self.lock:
            self.face_detected = bool(result and result.multi_face_landmarks)
            if not self.face_detected:
                self.unknown_seconds += dt
                if now-self.last_log_time > 2:
                    self.log("FACE_NOT_DETECTED", "No face detected")
                    self.last_log_time = now
                return av.VideoFrame.from_ndarray(image, format="bgr24")

            lm = result.multi_face_landmarks[0].landmark
            yaw, pitch = head_pose(lm, w, h)
            ear = (eye_aspect_ratio(lm, LEFT_EYE, w, h) + eye_aspect_ratio(lm, RIGHT_EYE, w, h))/2
            gx, gy = gaze_values(lm)
            self.last_yaw, self.last_pitch, self.last_ear = yaw, pitch, ear

            if self.calibrate_requested:
                if yaw is not None and pitch is not None:
                    self.cyaw.append(yaw); self.cpitch.append(pitch)
                if gx is not None and gy is not None:
                    self.cgx.append(gx); self.cgy.append(gy)
                if len(self.cyaw) >= CALIBRATION_HEAD_SAMPLES and len(self.cgx) >= CALIBRATION_GAZE_SAMPLES:
                    self.nyaw, self.npitch = float(np.median(self.cyaw)), float(np.median(self.cpitch))
                    self.ngx, self.ngy = float(np.median(self.cgx)), float(np.median(self.cgy))
                    self.calibrated, self.calibrate_requested = True, False
                    self.log("CALIBRATION_COMPLETED", "Head and gaze baseline stored")

            direction = "Unknown"
            if yaw is not None and pitch is not None:
                if self.calibrated:
                    dy, dp = yaw-self.nyaw, pitch-self.npitch
                    direction = "Right" if dy > HEAD_YAW_THRESHOLD else "Left" if dy < -HEAD_YAW_THRESHOLD else "Down" if dp > HEAD_PITCH_THRESHOLD else "Up" if dp < -HEAD_PITCH_THRESHOLD else "Center"
                else:
                    direction = "Uncalibrated"
            gaze = "Unknown"
            if gx is not None and gy is not None and self.calibrated:
                gaze = "Away" if abs(gx-self.ngx) > GAZE_THRESHOLD or abs(gy-self.ngy) > GAZE_THRESHOLD else "Center"

            lookaway = direction in ("Left", "Right", "Up", "Down") or gaze == "Away"
            if self.calibrated:
                if direction == "Center" and gaze in ("Center", "Unknown"):
                    self.center_seconds += dt
                elif lookaway:
                    self.lookaway_seconds += dt
                else:
                    self.unknown_seconds += dt
            else:
                self.unknown_seconds += dt

            if direction in self.direction_counts:
                self.direction_counts[direction] += 1

            if lookaway:
                if self.lookaway_since is None:
                    self.lookaway_since = now
                if not self.was_lookaway and now-self.lookaway_since >= LOOK_AWAY_WARNING_SECONDS:
                    self.lookaways += 1
                    self.notes.append("LOOK_AWAY_WARNING")
                    self.log("LOOK_AWAY_WARNING", f"Head={direction}; gaze={gaze}")
            else:
                self.lookaway_since = None
            self.was_lookaway = lookaway

            closed = ear < EAR_CLOSED_THRESHOLD
            if closed:
                if self.eye_closed_since is None:
                    self.eye_closed_since = now
                self.closed_seconds += dt
                if now-self.eye_closed_since >= EYE_CLOSED_WARNING_SECONDS and not self.was_eye_closed:
                    self.eye_warnings += 1
                    self.notes.append("EYE_CLOSURE_WARNING")
                    self.log("EYE_CLOSURE_WARNING", f"EAR={ear:.3f}")
                    self.was_eye_closed = True
            else:
                if self.eye_closed_since is not None:
                    elapsed = now-self.eye_closed_since
                    if 0.08 <= elapsed < EYE_CLOSED_WARNING_SECONDS:
                        self.blinks += 1
                        self.log("BLINK", f"Closed duration={elapsed:.2f}s")
                self.eye_closed_since = None
                self.was_eye_closed = False

            self.last_direction, self.last_gaze = direction, gaze
            if now-self.last_log_time >= 5:
                self.log("SIGNAL_SAMPLE", f"head={direction}; gaze={gaze}; EAR={ear:.3f}; calibrated={self.calibrated}")
                self.last_log_time = now

        # Do not draw landmarks, lines, circles, boxes, or text on the camera image.
        return av.VideoFrame.from_ndarray(image, format="bgr24")

    def close(self):
        try:
            self.face_mesh.close()
        except Exception:
            pass


def save_session(proc):
    summary = proc.snapshot()
    events = list(proc.events)
    e_ok, e_msg = append_csv(EVENT_PATH, events, EVENT_COLUMNS)
    s_ok, s_msg = append_csv(SUMMARY_PATH, [summary], SUMMARY_COLUMNS)
    return e_ok and s_ok, summary["Session_ID"], e_msg, s_msg


st.title("NeuroAdaptiveInterface")
st.caption("Camera-based gaze, head direction and eye-closure signal demo — not a medical diagnosis.")
with st.sidebar:
    st.header("Settings")
    st.write("Camera preview: clean, with no face marks or overlay text.")
    st.write(f"Look-away warning: {LOOK_AWAY_WARNING_SECONDS:.0f} seconds")
    st.write(f"Eye-closure warning: {EYE_CLOSED_WARNING_SECONDS:.0f} seconds")
    token, repo, branch = github_config()
    if token and repo:
        st.success(f"GitHub storage configured: {repo} ({branch})")
    else:
        st.info("GitHub Secrets are not set; logs will be stored locally.")

st.info("Allow camera access. Sit in front of the camera, look straight, then calibrate.")
ctx = webrtc_streamer(
    key="neuroadaptive-face-demo",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=FaceProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
proc = ctx.video_processor

if proc is not None:
    a, b = st.columns(2)
    if a.button("🎯 Calibrate / Recalibrate", use_container_width=True):
        proc.request_calibration()
        st.success("Calibration started. Look at the camera and keep your head steady.")
    if b.button("↩️ Reset calibration", use_container_width=True):
        proc.reset_calibration()
        st.warning("Calibration reset. Calibrate again before using gaze/head direction.")

    data = proc.snapshot()
    st.subheader("Live session")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{data['Duration_seconds']:.1f} sec")
    c2.metric("Blinks", data["Blink_Count"])
    c3.metric("Look-away episodes", data["Look_Away_Count"])
    c4.metric("Eye-closure warnings", data["Eye_Closure_Warning_Count"])
    c5, c6, c7 = st.columns(3)
    c5.metric("Head direction", proc.last_direction)
    c6.metric("Gaze signal", proc.last_gaze)
    c7.metric("Calibration", "Ready" if proc.calibrated else "Collecting…" if proc.calibrate_requested else "Not calibrated")

    with st.expander("Detailed signal values"):
        st.write({
            "Face detected": proc.face_detected,
            "Estimated head yaw": None if proc.last_yaw is None else round(proc.last_yaw, 2),
            "Estimated head pitch": None if proc.last_pitch is None else round(proc.last_pitch, 2),
            "Eye aspect ratio (EAR)": None if proc.last_ear is None else round(proc.last_ear, 3),
            "Center gaze seconds": round(proc.center_seconds, 2),
            "Look-away seconds": round(proc.lookaway_seconds, 2),
            "Unknown signal seconds": round(proc.unknown_seconds, 2),
            "Eyes-closed seconds": round(proc.closed_seconds, 2),
            "Head direction sample counts": proc.direction_counts,
        })

    if not ctx.state.playing:
        if st.button("💾 Save session to CSV / GitHub", type="primary", use_container_width=True):
            saved_key = f"saved_{proc.session_id}"
            if st.session_state.get(saved_key):
                st.warning("This session has already been saved.")
            else:
                with st.spinner("Saving session…"):
                    ok, sid, emsg, smsg = save_session(proc)
                if ok:
                    st.session_state[saved_key] = True
                    st.success(f"Session {sid} saved. {emsg} {smsg}")
                else:
                    st.error(f"Save incomplete. Events: {emsg} Summary: {smsg}")
    else:
        st.warning("Stop the camera first, then save the session.")
else:
    st.warning("Camera processor is starting. Wait a few seconds and allow camera permission.")

st.divider()
st.subheader("Saved session history")
events_df = load_csv(EVENT_PATH, EVENT_COLUMNS)
summary_df = load_csv(SUMMARY_PATH, SUMMARY_COLUMNS)
tab1, tab2 = st.tabs(["Session summaries", "Event log"])
with tab1:
    if summary_df.empty:
        st.info("No saved session summaries yet.")
    else:
        st.dataframe(summary_df.tail(100), use_container_width=True, hide_index=True)
        st.download_button("Download session_summary.csv", summary_df.to_csv(index=False).encode("utf-8"), "session_summary.csv", "text/csv")
with tab2:
    if events_df.empty:
        st.info("No saved events yet.")
    else:
        st.dataframe(events_df.tail(300), use_container_width=True, hide_index=True)
        st.download_button("Download event_log.csv", events_df.to_csv(index=False).encode("utf-8"), "event_log.csv", "text/csv")

st.caption("Video frames are not uploaded by this code. Only CSV logs are sent to GitHub when GitHub Secrets are configured.")
