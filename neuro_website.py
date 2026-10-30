import os
import time

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Keep this module in the same GitHub folder as neuro_website.py.
from webcam_processor import NeuroVideoProcessor


# ---------------------------------------------------------
# APP SETTINGS
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "session_logs")
SUMMARY_FILE = os.path.join(LOG_DIR, "session_summary.csv")
EVENT_FILE = os.path.join(LOG_DIR, "event_log.csv")

os.makedirs(LOG_DIR, exist_ok=True)

st.set_page_config(
    page_title="NeuroAdaptiveInterface",
    page_icon="🧠",
    layout="wide"
)


# ---------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------
def read_csv_safely(path):
    """Read a CSV without stopping the whole dashboard if it is missing."""
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception as exc:
        st.warning(
            "Could not read {}: {}".format(os.path.basename(path), exc)
        )
        return pd.DataFrame()


def load_summary():
    df = read_csv_safely(SUMMARY_FILE)
    if not df.empty and "Session_ID" in df.columns:
        df = df.drop_duplicates(subset=["Session_ID"], keep="last")
    return df.reset_index(drop=True)


def load_events():
    return read_csv_safely(EVENT_FILE)


def number(value, default=0.0):
    """Convert a CSV value to a safe number."""
    result = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(result) else float(result)


def show_session_details(session, available_columns):
    st.subheader("📌 Selected Session Details")

    c1, c2, c3 = st.columns(3)
    c1.metric("Duration", "{:.2f} sec".format(
        number(session.get("duration_seconds", 0))
    ))
    c2.metric("Blink Count", int(number(session.get("blink_count", 0))))
    c3.metric("Look-away Count", int(number(session.get("look_away_count", 0))))

    gaze_fields = {
        "Center Gaze (sec)": "center_gaze_seconds",
        "Look-away (sec)": "look_away_seconds",
        "Unknown Gaze (sec)": "unknown_gaze_seconds",
        "Eyes Closed (sec)": "eyes_closed_seconds"
    }

    gaze_rows = []
    for label, field in gaze_fields.items():
        if field in available_columns:
            gaze_rows.append({
                "Signal": label,
                "Duration (sec)": round(number(session.get(field, 0)), 2)
            })

    if gaze_rows:
        st.subheader("👁️ Gaze Signal Summary")
        st.dataframe(
            pd.DataFrame(gaze_rows),
            use_container_width=True,
            hide_index=True
        )

    with st.expander("📄 Complete Session Record"):
        details = session.to_frame(name="Value")
        details.index.name = "Field"
        st.dataframe(details, use_container_width=True)


# ---------------------------------------------------------
# HEADER AND NAVIGATION
# ---------------------------------------------------------
st.title("🧠 NeuroAdaptiveInterface")
st.caption(
    "Experimental facial-signal monitoring and session analytics"
)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Home", "Session Dashboard", "Session History", "Event Log"]
)


# ---------------------------------------------------------
# HOME - BROWSER CAMERA
# ---------------------------------------------------------
if page == "Home":
    st.header("🎥 Live Camera Monitoring")

    st.info(
        "Allow camera permission in your browser. Look straight at the "
        "camera while calibration collects samples."
    )

    # Use one STUN server. Do not configure TURN unless you have TURN
    # credentials from a provider.
    rtc_configuration = {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }

    ctx = webrtc_streamer(
        key="neuroadaptive-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=NeuroVideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    st.caption(
        "Camera processing runs through the browser video stream. "
        "No separate OpenCV window is required."
    )

    processor = ctx.video_processor

    if processor is not None:
        # Read values under the processor lock because the WebRTC callback
        # updates them in a separate worker thread.
        with processor.lock:
            face_detected = processor.face_detected
            gaze = processor.gaze
            eye_status = processor.eye_status
            blinks = processor.blinks
            head = processor.head
            calibrated = processor.calibrated
            message = processor.message
            started_at = processor.started_at

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Face", "Detected" if face_detected else "Not detected")
        c2.metric("Gaze", str(gaze))
        c3.metric("Eye status", str(eye_status))
        c4.metric("Blinks", int(blinks))

        st.write("**Head direction:**", head)
        st.write(
            "**Calibration:**",
            "Complete" if calibrated else "Collecting samples"
        )
        st.write("**Status:**", message)
        st.write(
            "**Session duration:** {:.1f} sec".format(
                max(0.0, time.time() - started_at)
            )
        )

        a, b = st.columns(2)
        with a:
            if st.button("🔄 Reset Calibration", use_container_width=True):
                processor.reset_calibration()
                st.success("Calibration reset requested.")
                st.rerun()

        with b:
            if st.button("💾 Save Session", use_container_width=True):
                processor.save_session()
                st.success(
                    "Save request sent. Check the Session Dashboard."
                )
                st.rerun()

    elif ctx.state.playing:
        st.info("Camera is connecting. Please wait.")
    else:
        st.info("Press START to begin camera monitoring.")

    st.divider()
    st.subheader("How to use")
    st.markdown(
        """
        1. Press **START** and allow camera access.
        2. Look straight at the camera while calibration collects samples.
        3. Move your eyes and head naturally to test tracking.
        4. Click **Save Session** before leaving the page.
        5. Review saved data in Session Dashboard or Session History.
        """
    )
    st.caption(
        "Experimental prototype only. Camera signals are not a medical "
        "diagnosis or a validated cognitive-overload measurement."
    )


# ---------------------------------------------------------
# SESSION DASHBOARD
# ---------------------------------------------------------
elif page == "Session Dashboard":
    st.header("📊 Session Dashboard")

    if st.button("↻ Refresh Dashboard"):
        st.rerun()

    df = load_summary()

    if df.empty:
        st.warning("No saved sessions found yet. Start a session and save it.")
    else:
        if "Session_ID" not in df.columns:
            st.error("Session_ID column is missing from session_summary.csv.")
        else:
            # Convert available metric columns to numeric values.
            numeric_columns = [
                "duration_seconds",
                "blink_count",
                "look_away_count",
                "center_gaze_seconds",
                "look_away_seconds",
                "unknown_gaze_seconds",
                "eyes_closed_seconds"
            ]
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            latest = df.iloc[-1]

            st.subheader("Latest Session Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Sessions", len(df))
            c2.metric(
                "Latest Duration",
                "{:.1f} sec".format(number(latest.get("duration_seconds", 0)))
            )
            c3.metric("Latest Blinks", int(number(latest.get("blink_count", 0))))
            c4.metric(
                "Look-away Warnings",
                int(number(latest.get("look_away_count", 0)))
            )

            st.subheader("Latest Session Record")
            st.dataframe(
                pd.DataFrame([latest]),
                use_container_width=True,
                hide_index=True
            )

            # Gaze time distribution for latest session.
            st.subheader("👁️ Gaze Time Distribution")
            gaze_labels = {
                "center_gaze_seconds": "Center Gaze",
                "look_away_seconds": "Look-away",
                "unknown_gaze_seconds": "Unknown Gaze",
                "eyes_closed_seconds": "Eyes Closed"
            }
            available_gaze = [
                col for col in gaze_labels if col in df.columns
            ]

            if available_gaze:
                gaze_chart = pd.DataFrame({
                    "Gaze Signal": [gaze_labels[col] for col in available_gaze],
                    "Time (seconds)": [
                        number(latest.get(col, 0)) for col in available_gaze
                    ]
                })
                fig = px.bar(
                    gaze_chart,
                    x="Time (seconds)",
                    y="Gaze Signal",
                    orientation="h",
                    text="Time (seconds)",
                    title="Latest Session Gaze Time"
                )
                fig.update_layout(
                    height=350,
                    yaxis_title="",
                    xaxis_title="Time (seconds)",
                    yaxis={"categoryorder": "total ascending"}
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(
                    gaze_chart,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Gaze time columns are not available in the CSV.")

            # Compare recent sessions.
            st.subheader("📊 Compare Sessions")
            compare_columns = [
                col for col in [
                    "duration_seconds", "blink_count", "look_away_count"
                ] if col in df.columns
            ]

            if compare_columns:
                compare_df = df.tail(10).copy()
                compare_df["Session"] = (
                    compare_df["Session_ID"].astype(str).str[-6:]
                )
                compare_df = compare_df.set_index("Session")
                selected_metrics = st.multiselect(
                    "Choose metrics to compare",
                    options=compare_columns,
                    default=compare_columns
                )
                if selected_metrics:
                    st.bar_chart(
                        compare_df[selected_metrics].fillna(0),
                        height=350,
                        use_container_width=True
                    )
                    st.dataframe(
                        compare_df[selected_metrics],
                        use_container_width=True
                    )
                else:
                    st.info("Select at least one metric.")
            else:
                st.info("Session comparison columns are not available.")

            # Session duration chart.
            st.subheader("Session Duration")
            if "duration_seconds" in df.columns:
                duration_df = df.tail(10).copy()
                duration_df["Session"] = (
                    duration_df["Session_ID"].astype(str).str[-6:]
                )
                duration_df["duration_seconds"] = pd.to_numeric(
                    duration_df["duration_seconds"], errors="coerce"
                )
                duration_df = duration_df.dropna(
                    subset=["duration_seconds"]
                ).set_index("Session")

                if not duration_df.empty:
                    st.bar_chart(
                        duration_df[["duration_seconds"]],
                        height=320,
                        use_container_width=True
                    )
                else:
                    st.info("No valid duration values found.")
            else:
                st.info("duration_seconds column is not available.")

            # Blink and look-away chart.
            st.subheader("Blink and Look-away Counts")
            count_columns = [
                col for col in ["blink_count", "look_away_count"]
                if col in df.columns
            ]
            if count_columns:
                count_df = df.tail(10).copy()
                count_df["Session"] = (
                    count_df["Session_ID"].astype(str).str[-6:]
                )
                for col in count_columns:
                    count_df[col] = pd.to_numeric(
                        count_df[col], errors="coerce"
                    ).fillna(0)

                plot_df = count_df.melt(
                    id_vars=["Session"],
                    value_vars=count_columns,
                    var_name="Signal",
                    value_name="Count"
                )
                fig = px.bar(
                    plot_df,
                    x="Count",
                    y="Session",
                    color="Signal",
                    barmode="group",
                    orientation="h",
                    text="Count"
                )
                fig.update_layout(
                    height=420,
                    xaxis_title="Count",
                    yaxis_title="Session ID"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Blink/look-away count columns are not available.")

            # Recent session table.
            st.subheader("Recent Session History")
            history_columns = [
                col for col in [
                    "Session_ID",
                    "duration_seconds",
                    "blink_count",
                    "look_away_count",
                    "eye_closure_warning_count",
                    "Calibrated",
                    "Last_Event"
                ] if col in df.columns
            ]
            if history_columns:
                st.dataframe(
                    df[history_columns].tail(10).iloc[::-1],
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                "These are recorded camera signals, not a medical diagnosis "
                "or a validated cognitive-overload measurement."
            )


# ---------------------------------------------------------
# SESSION HISTORY
# ---------------------------------------------------------
elif page == "Session History":
    st.header("📁 Session History")
    df = load_summary()

    if df.empty:
        st.info("No saved sessions found.")
    elif "Session_ID" not in df.columns:
        st.error("Session_ID column is missing from session_summary.csv.")
    else:
        history = df.iloc[::-1].reset_index(drop=True)

        st.subheader("All Saved Sessions")
        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True
        )

        session_ids = history["Session_ID"].astype(str).tolist()
        selected = st.selectbox(
            "Select a session to view details",
            session_ids
        )
        match = history[history["Session_ID"].astype(str) == selected]

        if not match.empty:
            show_session_details(match.iloc[0], history.columns)


# ---------------------------------------------------------
# EVENT LOG
# ---------------------------------------------------------
elif page == "Event Log":
    st.header("📝 Event Log")
    events = load_events()

    if events.empty:
        st.info("No events recorded yet.")
    else:
        events = events.copy()
        st.subheader("🔍 Filter Events")

        col1, col2 = st.columns(2)
        with col1:
            if "Session_ID" in events.columns:
                session_ids = (
                    events["Session_ID"].dropna().astype(str).unique().tolist()
                )
                selected_session = st.selectbox(
                    "Filter by session",
                    ["All Sessions"] + sorted(session_ids, reverse=True)
                )
            else:
                selected_session = "All Sessions"

        with col2:
            if "Event" in events.columns:
                event_types = (
                    events["Event"].dropna().astype(str).unique().tolist()
                )
                selected_event = st.selectbox(
                    "Filter by event type",
                    ["All Events"] + sorted(event_types)
                )
            else:
                selected_event = "All Events"

        show_important_only = st.checkbox(
            "⭐ Show important events only",
            value=False
        )
        important_events = [
            "BLINK",
            "LOOK_AWAY_WARNING",
            "EYE_CLOSURE_WARNING",
            "SESSION_STARTED",
            "SESSION_ENDED",
            "CALIBRATION_COMPLETED"
        ]

        if (
            selected_session != "All Sessions"
            and "Session_ID" in events.columns
        ):
            events = events[
                events["Session_ID"].astype(str) == selected_session
            ]

        if selected_event != "All Events" and "Event" in events.columns:
            events = events[events["Event"].astype(str) == selected_event]

        if show_important_only and "Event" in events.columns:
            events = events[
                events["Event"].astype(str).str.upper().isin(important_events)
            ]

        events = events.iloc[::-1].reset_index(drop=True)
        st.divider()
        st.subheader("📊 Event Summary")

        event_names = (
            events["Event"].astype(str).str.upper()
            if "Event" in events.columns
            else pd.Series(dtype=str)
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Events", len(events))
        c2.metric(
            "Look-away Warnings",
            int((event_names == "LOOK_AWAY_WARNING").sum())
        )
        c3.metric("Blink Events", int((event_names == "BLINK").sum()))
        c4.metric(
            "Session Ended",
            int((event_names == "SESSION_ENDED").sum())
        )

        if events.empty:
            st.warning("No events match the selected filters.")
        else:
            st.dataframe(
                events,
                use_container_width=True,
                hide_index=True
            )
            st.download_button(
                label="⬇️ Download Events as CSV",
                data=events.to_csv(index=False).encode("utf-8"),
                file_name="neuro_event_log.csv",
                mime="text/csv"
            )


st.divider()
st.caption("NeuroAdaptiveInterface · Experimental research prototype")
