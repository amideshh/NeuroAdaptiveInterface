
import os
import time
import csv
from datetime import datetime
from collections import deque
from streamlit_autorefresh import st_autorefresh
import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode
)
st_autorefresh(
    interval=1000,
    key="camera_metrics_refresh"
    )
# ---------------------------------------
# SETTINGS
# ---------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_DEMO = os.path.join(BASE_DIR, "face_demo.py")
LOG_DIR = os.path.join(BASE_DIR, "session_logs")

SUMMARY_FILE = os.path.join(LOG_DIR, "session_summary.csv")
EVENT_FILE = os.path.join(LOG_DIR, "event_log.csv")

os.makedirs(LOG_DIR, exist_ok=True)

st.set_page_config(
    page_title="NeuroAdaptiveInterface",
    page_icon="🧠",
    layout="wide"
)

if "camera_process" not in st.session_state:
    st.session_state.camera_process = None


# ---------------------------------------
# CSV HELPERS
# ---------------------------------------

def read_csv_safely(path):
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception as exc:
        st.warning("Could not read {}: {}".format(
            os.path.basename(path), exc
        ))
        return pd.DataFrame()


def load_summary():
    return read_csv_safely(SUMMARY_FILE)


def load_events():
    return read_csv_safely(EVENT_FILE)


# ---------------------------------------
# HEADER
# ---------------------------------------

st.title("🧠 NeuroAdaptiveInterface")
st.caption(
    "Experimental facial-signal monitoring and session analytics"
)

st.info(
    "Camera processing runs in a separate OpenCV window. "
    "Use this website to start monitoring and review saved sessions."
)

# ---------------------------------------
# SIDEBAR
# ---------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Home",
        "Session Dashboard",
        "Session History",
        "Event Log"
    ]
)


# ---------------------------------------
# HOME PAGE - BROWSER CAMERA
# ---------------------------------------

if page == "Home":

    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    from webcam_processor import NeuroVideoProcessor

    st.header("🎥 Live Camera Monitoring")

    st.info(
        "Allow camera permission in your browser. "
        "Look straight at the camera while calibration collects samples."
    )

    ctx = webrtc_streamer(
        key="neuroadaptive-camera",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=NeuroVideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

    st.caption(
        "Camera processing happens in your browser video stream. "
        "No separate OpenCV window is required."
    )

    if ctx.video_processor:
        processor = ctx.video_processor

        with processor.lock:
            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Face", "Detected" if processor.face_detected else "Not detected")
            c2.metric("Gaze", processor.gaze)
            c3.metric("Eye status", processor.eye_status)
            c4.metric("Blinks", processor.blinks)

            st.write("**Head direction:**", processor.head)
            st.write("**Calibration:**", "Complete" if processor.calibrated else "Collecting samples")
            st.write("**Status:**", processor.message)

            st.write("**Session duration:** {:.1f} sec".format(
                time.time() - processor.started_at
            ))

            a, b = st.columns(2)

            with a:
                if st.button("🔄 Reset Calibration", use_container_width=True):
                    processor.reset_calibration()
                    st.rerun()

            with b:
                if st.button("💾 Save Session", use_container_width=True):
                    processor.save_session()
                    st.success("Session save requested. Check Session Dashboard.")
                    st.rerun()

    st.divider()

    st.subheader("How to use")
    st.markdown(
        """
        1. Start the camera above.
        2. Allow camera access when your browser asks.
        3. Look straight at the camera while calibration collects 20 samples.
        4. Move your eyes and head naturally to test tracking.
        5. Click **Save Session** before leaving the page.
        6. Review saved data in Session Dashboard or Session History.
        """
    )

    st.caption(
        "Experimental prototype only. Signals are not a medical diagnosis."
    )


# ---------------------------------------
# SESSION DASHBOARD
# ---------------------------------------


elif page == "Session Dashboard":

    st.header("📊 Session Dashboard")

    if st.button("Refresh Dashboard"):
        st.rerun()

    df = load_summary()

    if df.empty:
        st.warning(
            "No session data found yet. "
            "Run a session and press Q to save it."
        )

    else:
        # Remove duplicate session records
        if "Session_ID" in df.columns:
            df = df.drop_duplicates(
                subset=["Session_ID"],
                keep="last"
            )

        # Convert session values to numeric
        for col in [
            "duration_seconds",
            "blink_count",
            "look_away_count"
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

        # Latest session
        latest = df.iloc[-1]

        # Dashboard metrics
        c1, c2, c3, c4 = st.columns(4)

        duration = pd.to_numeric(
            pd.Series([latest.get("duration_seconds", 0)]),
            errors="coerce"
        ).fillna(0).iloc[0]

        blink = pd.to_numeric(
            pd.Series([latest.get("blink_count", 0)]),
            errors="coerce"
        ).fillna(0).iloc[0]

        lookaway = pd.to_numeric(
            pd.Series([latest.get("look_away_count", 0)]),
            errors="coerce"
        ).fillna(0).iloc[0]

        c1.metric("Total Sessions", len(df))
        c2.metric(
            "Latest Duration",
            "{:.1f} sec".format(duration)
        )
        c3.metric("Latest Blinks", int(blink))
        c4.metric("Look-away Warnings", int(lookaway))

        # Latest session details
        st.subheader("Latest Session")

        st.dataframe(
            pd.DataFrame([latest]),
            use_container_width=True,
            hide_index=True
        )



        # Gaze Time Distribution
        st.subheader("👁️ Gaze Time Distribution")

        gaze_columns = [
            "center_gaze_seconds",
            "look_away_seconds",
            "unknown_gaze_seconds",
            "eyes_closed_seconds"
        ]

        available_gaze_columns = [
            col for col in gaze_columns
            if col in df.columns
        ]

        if available_gaze_columns:

            gaze_data = latest[available_gaze_columns]

            gaze_data = pd.to_numeric(
                gaze_data,
                errors="coerce"
            ).fillna(0)

            gaze_labels = {
                "center_gaze_seconds": "Center Gaze",
                "look_away_seconds": "Look-away",
                "unknown_gaze_seconds": "Unknown Gaze",
                "eyes_closed_seconds": "Eyes Closed"
            }

            gaze_chart = pd.DataFrame({
                "Gaze Signal": [
                    gaze_labels[col]
                    for col in available_gaze_columns
                ],
                "Time (seconds)": [
                    float(gaze_data[col])
                    for col in available_gaze_columns
                ]
            })

            fig = px.bar(
                gaze_chart,
                x="Time (seconds)",
                y="Gaze Signal",
                orientation="h",
                text="Time (seconds)",
                title="Gaze Time Distribution"
            )

            fig.update_layout(
                height=350,
                yaxis_title="",
                xaxis_title="Time (seconds)",
                yaxis={"categoryorder": "total ascending"}
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                gaze_chart,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("Gaze time data is not available.")



        # Compare Sessions
        st.subheader("📊 Compare Sessions")

        compare_columns = [
            col for col in [
                "duration_seconds",
                "blink_count",
                "look_away_count"
            ]
            if col in df.columns
        ]

        if "Session_ID" in df.columns and compare_columns:
            compare_df = df.drop_duplicates(
                subset=["Session_ID"],
                keep="last"
            ).tail(10).copy()

            for col in compare_columns:
                compare_df[col] = pd.to_numeric(
                    compare_df[col],
                    errors="coerce"
                )

            compare_df["Session"] = (
                compare_df["Session_ID"]
                .astype(str)
                .str[-6:]
            )

            compare_df = compare_df.set_index("Session")

            st.caption(
                "Compare the latest 10 sessions using recorded signals."
            )

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
                st.info("Select at least one metric to compare.")

        else:
            st.info("Session comparison data is not available.")







        # Session duration chart

        st.subheader("Session Duration")

        chart_df = df.tail(10).copy()

        chart_df["duration_seconds"] = pd.to_numeric(
            chart_df["duration_seconds"],
            errors="coerce"
        )

        chart_df["Session"] = (
            chart_df["Session_ID"].astype(str).str[-6:]
        )

        chart_df = chart_df.dropna(
            subset=["duration_seconds"]
        ).set_index("Session")

        st.write("Duration chart rows:", len(chart_df))
        st.dataframe(chart_df[["duration_seconds"]])

        if not chart_df.empty:
            st.bar_chart(
                data=chart_df,
                y="duration_seconds",
                height=350,
                use_container_width=True
            )
        else:
            st.warning("No valid duration values found.")


        # Blink and look-away chart

        st.subheader("Blink and Look-away Counts")

        count_columns = [
            col for col in [
                "blink_count",
                "look_away_count"
            ]
            if col in df.columns
        ]

        if count_columns:
            count_df = df.tail(10).copy()

            for col in count_columns:
                count_df[col] = pd.to_numeric(
                    count_df[col],
                    errors="coerce"
                )

            count_df["Session"] = (
                count_df["Session_ID"]
                .astype(str)
                .str[-6:]
            )

            count_df = count_df.set_index("Session")


            if count_columns and "Session_ID" in df.columns:
                count_df = df.tail(10).copy()

                for col in count_columns:
                    count_df[col] = pd.to_numeric(
                        count_df[col],
                        errors="coerce"
                    ).fillna(0)

                count_df["Session"] = (
                    count_df["Session_ID"]
                    .astype(str)
                    .str[-6:]
                )

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
                    height=450,
                    xaxis_title="Count",
                    yaxis_title="Session ID"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:
                st.info(
                    "Blink and look-away data is not available."
                )



        # Recent session history
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
            ]
            if col in df.columns
        ]

        if history_columns:
            st.dataframe(
                df[history_columns].tail(10).iloc[::-1],
                use_container_width=True,
                hide_index=True
            )

        # Disclaimer
        st.caption(
            "These are recorded camera signals, not a medical "
            "diagnosis or a validated cognitive-overload measurement."
        )

# ---------------------------------------
# SESSION HISTORY
# ---------------------------------------

elif page == "Session History":

    st.header("📁 Session History")

    df = load_summary()

    if df.empty:
        st.info("No saved sessions found.")
    else:
        if "Session_ID" in df.columns:
            df = df.drop_duplicates(
                subset=["Session_ID"],
                keep="last"
            )

        st.dataframe(
            df.iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        if "Session_ID" in df.columns:
            selected = st.selectbox(
                "Select a session",
                df["Session_ID"].astype(str).tolist()[::-1]
            )

            match = df[
                df["Session_ID"].astype(str) == selected
            ]

            if not match.empty:
                session = match.iloc[0]

                st.subheader("📌 Selected Session Details")

                col1, col2, col3 = st.columns(3)

                duration = pd.to_numeric(
                    session.get("duration_seconds", 0),
                    errors="coerce"
                )
                blinks = pd.to_numeric(
                    session.get("blink_count", 0),
                    errors="coerce"
                )
                lookaways = pd.to_numeric(
                    session.get("look_away_count", 0),
                    errors="coerce"
                )

                with col1:
                    st.metric(
                        "Duration",
                        "{:.2f} sec".format(
                            0 if pd.isna(duration) else duration
                        )
                    )

                with col2:
                    st.metric(
                        "Blink Count",
                        int(0 if pd.isna(blinks) else blinks)
                    )

                with col3:
                    st.metric(
                        "Look-away Count",
                        int(0 if pd.isna(lookaways) else lookaways)
                    )

                st.subheader("👁️ Gaze Signal Summary")

                gaze_fields = {
                    "Center Gaze (sec)": "center_gaze_seconds",
                    "Look-away (sec)": "look_away_seconds",
                    "Unknown Gaze (sec)": "unknown_gaze_seconds",
                    "Eyes Closed (sec)": "eyes_closed_seconds"
                }

                gaze_rows = []

                for label, field in gaze_fields.items():
                    if field in df.columns:
                        value = pd.to_numeric(
                            session.get(field, 0),
                            errors="coerce"
                        )

                        gaze_rows.append({
                            "Signal": label,
                            "Duration (sec)": round(
                                0 if pd.isna(value) else float(value),
                                2
                            )
                        })

                if gaze_rows:
                    st.dataframe(
                        pd.DataFrame(gaze_rows),
                        use_container_width=True,
                        hide_index=True
                    )

                with st.expander("📄 Complete Session Record"):
                    details = session.to_frame(name="Value")
                    details.index.name = "Field"

                    st.dataframe(
                        details,
                        use_container_width=True
                    )


            # ---------------------------------------
            # SESSION HISTORY
            # ---------------------------------------

            elif page == "Session History":

                st.header("📁 Session History")

                df = load_summary()

                if df.empty:
                    st.info("No saved sessions found.")

                else:
                    if "Session_ID" in df.columns:
                        df = df.drop_duplicates(
                            subset=["Session_ID"],
                            keep="last"
                        ).copy()

                        df = df.iloc[::-1].reset_index(drop=True)

                    st.subheader("All Saved Sessions")

                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )

                    if "Session_ID" in df.columns:

                        session_ids = df["Session_ID"].astype(str).tolist()

                        selected = st.selectbox(
                            "Select a session to view details",
                            session_ids
                        )

                        match = df[
                            df["Session_ID"].astype(str) == selected
                            ]

                        if not match.empty:

                            session = match.iloc[0]

                            st.divider()
                            st.subheader("📌 Selected Session Details")

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                duration = pd.to_numeric(
                                    session.get("duration_seconds", 0),
                                    errors="coerce"
                                )
                                st.metric(
                                    "Duration",
                                    "{:.2f} sec".format(
                                        0 if pd.isna(duration) else duration
                                    )
                                )

                            with col2:
                                blinks = pd.to_numeric(
                                    session.get("blink_count", 0),
                                    errors="coerce"
                                )
                                st.metric(
                                    "Blink Count",
                                    int(0 if pd.isna(blinks) else blinks)
                                )

                            with col3:
                                lookaways = pd.to_numeric(
                                    session.get("look_away_count", 0),
                                    errors="coerce"
                                )
                                st.metric(
                                    "Look-away Count",
                                    int(0 if pd.isna(lookaways) else lookaways)
                                )

                            st.subheader("👁️ Gaze Signal Summary")

                            gaze_fields = {
                                "Center Gaze (sec)": "center_gaze_seconds",
                                "Look-away (sec)": "look_away_seconds",
                                "Unknown Gaze (sec)": "unknown_gaze_seconds",
                                "Eyes Closed (sec)": "eyes_closed_seconds"
                            }

                            gaze_rows = []

                            for label, field in gaze_fields.items():
                                if field in df.columns:
                                    value = pd.to_numeric(
                                        session.get(field, 0),
                                        errors="coerce"
                                    )

                                    gaze_rows.append({
                                        "Signal": label,
                                        "Duration (sec)": round(
                                            0 if pd.isna(value) else float(value),
                                            2
                                        )
                                    })

                            if gaze_rows:
                                st.dataframe(
                                    pd.DataFrame(gaze_rows),
                                    use_container_width=True,
                                    hide_index=True
                                )

                            st.subheader("📄 Complete Session Record")

                            details = session.to_frame(name="Value")
                            details.index.name = "Field"

                            st.dataframe(
                                details,
                                use_container_width=True
                            )

                    else:
                        st.warning(
                            "Session_ID column is missing from session summary data."
                        )


# ---------------------------------------
# EVENT LOG
# ---------------------------------------

elif page == "Event Log":

    st.header("📝 Event Log")

    events = load_events()

    if events.empty:
        st.info("No events recorded yet.")

    else:
        events = events.copy()

        # Filters
        st.subheader("🔍 Filter Events")

        col1, col2 = st.columns(2)

        with col1:
            if "Session_ID" in events.columns:
                session_ids = (
                    events["Session_ID"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                selected_session = st.selectbox(
                    "Filter by session",
                    ["All Sessions"] + sorted(
                        session_ids,
                        reverse=True
                    )
                )
            else:
                selected_session = "All Sessions"

        with col2:
            if "Event" in events.columns:
                event_types = (
                    events["Event"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                selected_event = st.selectbox(
                    "Filter by event type",
                    ["All Events"] + sorted(event_types)
                )
            else:
                selected_event = "All Events"

            # Important Events Filter
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


        # Apply filters
        if selected_session != "All Sessions":
            events = events[
                events["Session_ID"].astype(str)
                == selected_session
            ]

        if selected_event != "All Events":
            events = events[
                events["Event"].astype(str)
                == selected_event
            ]

        if show_important_only and "Event" in events.columns:
            events = events[
                events["Event"].astype(str).str.upper().isin(
                    important_events
                )
            ]

        # Latest events first
        events = events.iloc[::-1].reset_index(drop=True)


        st.divider()


        # ---------------------------------------
        # EVENT SUMMARY CARDS
        # ---------------------------------------

        st.subheader("📊 Event Summary")

        event_col1, event_col2, event_col3, event_col4 = (
            st.columns(4)
        )

        # Count events after applying filters
        event_names = (
            events["Event"].astype(str).str.upper()
            if "Event" in events.columns
            else pd.Series(dtype=str)
        )

        with event_col1:
            st.metric(
                "Total Events",
                len(events)
            )

        with event_col2:
            st.metric(
                "Look-away Warnings",
                int((event_names == "LOOK_AWAY_WARNING").sum())
            )

        with event_col3:
            st.metric(
                "Blink Events",
                int((event_names == "BLINK").sum())
            )

        with event_col4:
            st.metric(
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

            # CSV download
            csv_data = events.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Download Events as CSV",
                data=csv_data,
                file_name="neuro_event_log.csv",
                mime="text/csv"
            )

st.divider()
st.caption(
    "NeuroAdaptiveInterface · Experimental research prototype"
)
