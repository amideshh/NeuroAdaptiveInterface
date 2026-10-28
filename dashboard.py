import os
import pandas as pd
def get_gaze_value(row, *column_names):
    for column in column_names:
        if column in row.index:
            value = pd.to_numeric(row[column], errors="coerce")
            if pd.notna(value):
                return float(value)

    return 0.0
import streamlit as st
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Neuro-Adaptive Analytics",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #f1f5f9;
}

h1, h2, h3 {
    color: #e2e8f0;
}

[data-testid="stHeader"] {
    background: rgba(11,17,32,.9);
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg,#172554,#1e293b);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 14px 16px;
    min-height: 115px;
    overflow: hidden;
}

div[data-testid="stMetricLabel"] {
    color: #cbd5e1;
    font-size: .9rem;
    white-space: normal;
    overflow-wrap: anywhere;
}

div[data-testid="stMetricValue"] {
    color: #38bdf8;
    font-size: clamp(1.2rem,2vw,2rem);
    white-space: normal;
    overflow-wrap: anywhere;
}

div.stButton > button,
div[data-testid="stDownloadButton"] > button {
    background: #0284c7;
    color: white;
    border: 0;
    border-radius: 9px;
    padding: 8px 16px;
}

div.stButton > button:hover,
div[data-testid="stDownloadButton"] > button:hover {
    background: #0369a1;
    color: white;
    border: 0;
}

div[data-testid="stTable"] {
    border: 1px solid #334155;
    border-radius: 10px;
    overflow: hidden;
}

div[data-baseweb="select"] > div {
    background: #1e293b;
    border-color: #475569;
    border-radius: 9px;
}

hr {
    border-color: #334155;
}

.dashboard-note {
    background: #111c30;
    border: 1px solid #26364f;
    border-radius: 10px;
    padding: 12px 16px;
    color: #cbd5e1;
    margin: 8px 0 18px 0;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# AUTO REFRESH
# --------------------------------------------------

st_autorefresh(
    interval=5000,
    limit=None,
    key="dashboard_auto_refresh"
)


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOG_DIR = os.path.join(BASE_DIR, "session_logs")

SUMMARY_FILE = os.path.join(
    LOG_DIR, "session_summary.csv"
)

EVENT_FILE = os.path.join(
    LOG_DIR, "event_log.csv"
)

ANALYSIS_FILE = os.path.join(
    LOG_DIR, "overload_analysis.csv"
)


# --------------------------------------------------
# SAFE CSV LOADER
# --------------------------------------------------

def load_csv(path):
    """Load a CSV safely."""

    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError
    ):
        return pd.DataFrame()


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

summary = load_csv(SUMMARY_FILE)
events = load_csv(EVENT_FILE)
analysis = load_csv(ANALYSIS_FILE)


# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------

st.title("🧠 Neuro-Adaptive Interface")
st.subheader("Live Analytics Dashboard")

st.caption(
    "Eye gaze, blink, look-away events and session history"
)

if st.button("🔄 Refresh Dashboard"):
    st.rerun()

st.markdown(
    '<div class="dashboard-note">'
    'Dashboard data refreshes automatically every 5 seconds. '
    'Your existing camera detection logic is not modified.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# CHECK SESSION DATA
# --------------------------------------------------

if summary.empty:
    st.warning(
        "No session data found. Run your existing camera "
        "program and complete a session first."
    )
    st.stop()

if "Session_ID" not in summary.columns:
    st.error(
        "Session_ID column is missing from session_summary.csv."
    )
    st.stop()


# --------------------------------------------------
# CLEAN SUMMARY DATA
# --------------------------------------------------

numeric_columns = [
    "duration_seconds",
    "blink_count",
    "look_away_count",
    "eye_closure_warning_count",
    "center_gaze_seconds",
    "look_away_seconds",
    "unknown_gaze_seconds",
    "eyes_closed_seconds"
]

for column in numeric_columns:
    if column in summary.columns:
        summary[column] = pd.to_numeric(
            summary[column],
            errors="coerce"
        ).fillna(0)

for column in ["session_start", "session_end"]:
    if column in summary.columns:
        summary[column] = pd.to_datetime(
            summary[column],
            errors="coerce"
        )

summary["Session_ID"] = (
    summary["Session_ID"].astype(str)
)

summary = summary.drop_duplicates(
    subset=["Session_ID"],
    keep="last"
)

session_ids = (
    summary["Session_ID"].dropna().tolist()
)

if not session_ids:
    st.warning("No valid Session_ID values were found.")
    st.stop()


# --------------------------------------------------
# SELECT SESSION
# --------------------------------------------------

selected_session = st.selectbox(
    "Select a session",
    options=session_ids[::-1]
)

session_rows = summary[
    summary["Session_ID"] == selected_session
]

if session_rows.empty:
    st.warning("Selected session was not found.")
    st.stop()

session = session_rows.iloc[-1]


# --------------------------------------------------
# SELECTED SESSION EVENTS
# --------------------------------------------------

if (
    not events.empty
    and "Session_ID" in events.columns
):
    events["Session_ID"] = (
        events["Session_ID"].astype(str)
    )

    session_events = events[
        events["Session_ID"] == selected_session
    ].copy()

else:
    session_events = pd.DataFrame()


# --------------------------------------------------
# SELECTED SESSION VALUES
# --------------------------------------------------

duration = float(
    session.get("duration_seconds", 0) or 0
)

blinks = int(
    float(session.get("blink_count", 0) or 0)
)

lookaways = int(
    float(session.get("look_away_count", 0) or 0)
)

eye_warnings = int(
    float(
        session.get("eye_closure_warning_count", 0) or 0
    )
)
center = float(
    session.get("Center_Seconds", 0) or 0
)

lookaway_time = float(
    session.get("Look_Away_Seconds", 0) or 0
)

unknown = float(
    session.get("Unknown_Gaze_Seconds", 0) or 0
)

closed_time = float(
    session.get("eyes_closed_seconds", 0) or 0
)


# --------------------------------------------------
# EVENT COUNTS
# --------------------------------------------------

clean_events = session_events.drop_duplicates().copy()

blink_count = blinks
look_away_count = lookaways
eye_warning_count = eye_warnings

if (
    not clean_events.empty
    and "Event" in clean_events.columns
):
    event_names = (
        clean_events["Event"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Use event-log counts when events are available.
    blink_count = int(
        (event_names == "BLINK").sum()
    )

    look_away_count = int(
        (event_names == "LOOK_AWAY_WARNING").sum()
    )

    eye_warning_names = [
        "EYE_CLOSURE_WARNING",
        "LONG_EYE_CLOSURE_WARNING",
        "EYES_CLOSED_WARNING",
        "EYE_CLOSED_WARNING"
    ]

    eye_warning_count = int(
        event_names.isin(eye_warning_names).sum()
    )


# --------------------------------------------------
# HEAD DIRECTION ANALYSIS
# --------------------------------------------------

st.divider()
st.subheader("🧭 Head Direction Analysis")

direction_order = [
    "LEFT", "RIGHT", "CENTER", "UP", "DOWN"
]

direction_counts = pd.Series(
    0,
    index=direction_order,
    dtype=int
)

if (
    not session_events.empty
    and "Details" in session_events.columns
):
    details = (
        session_events["Details"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    # Extract direction from text such as:
    # "Head direction: Right"

    extracted = details.str.extract(
        r"HEAD DIRECTION:\s*(LEFT|RIGHT|CENTER|UP|DOWN)",
        expand=False
    ).dropna()

    direction_counts = (
        extracted.value_counts()
        .reindex(direction_order, fill_value=0)
    )

    # Also support explicit HEAD_DIRECTION event rows.
    if "Event" in session_events.columns:
        head_events = session_events[
            session_events["Event"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
            .eq("HEAD_DIRECTION")
        ]

        if not head_events.empty:
            explicit = (
                head_events["Details"]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.extract(
                    r"\b(LEFT|RIGHT|CENTER|UP|DOWN)\b",
                    expand=False
                )
                .dropna()
            )

            direction_counts = (
                direction_counts.add(
                    explicit.value_counts(),
                    fill_value=0
                )
                .reindex(direction_order, fill_value=0)
                .astype(int)
            )


c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.metric(
        "⬅️ Left",
        int(direction_counts["LEFT"])
    )

with c2:
    st.metric(
        "➡️ Right",
        int(direction_counts["RIGHT"])
    )

with c3:
    st.metric(
        "⬆️ Center",
        int(direction_counts["CENTER"])
    )

with c4:
    st.metric(
        "🔼 Up",
        int(direction_counts["UP"])
    )

with c5:
    st.metric(
        "🔽 Down",
        int(direction_counts["DOWN"])
    )

if int(direction_counts.sum()) > 0:
    st.bar_chart(direction_counts)

st.caption(
    "Counts represent direction mentions in logged events, "
    "not the total time spent looking in each direction."
)


# --------------------------------------------------
# SELECTED SESSION OVERVIEW
# --------------------------------------------------

st.divider()
st.subheader("📌 Selected Session Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "⏱️ Duration (sec)",
        round(duration, 1)
    )

with col2:
    st.metric(
        "👁️ Blinks",
        blink_count
    )

with col3:
    st.metric(
        "↗️ Look-away",
        look_away_count
    )

with col4:
    st.metric(
        "⚠️ Eye warnings",
        eye_warning_count
    )


# --------------------------------------------------
# GAZE TIME DISTRIBUTION
# --------------------------------------------------

st.divider()
st.subheader("📊 Gaze Time Distribution")

gaze_data = pd.DataFrame({
    "Gaze state": [
        "Center gaze",
        "Look-away gaze",
        "Unknown gaze"
    ],
    "Seconds": [
        center,
        lookaway_time,
        unknown
    ]
})

left, right = st.columns(2)

with left:
    fig = px.pie(
        gaze_data,
        names="Gaze state",
        values="Seconds",
        title="Gaze Time Distribution",
        hole=0.45,
        color="Gaze state",
        color_discrete_map={
            "Center gaze": "#38bdf8",
            "Look-away gaze": "#f59e0b",
            "Unknown gaze": "#94a3b8"
        }
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        title_font_color="#e2e8f0",
        margin=dict(l=10, r=10, t=55, b=10)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:
    fig = px.bar(
        gaze_data,
        x="Gaze state",
        y="Seconds",
        color="Gaze state",
        title="Time by Gaze State",
        text="Seconds",
        color_discrete_map={
            "Center gaze": "#38bdf8",
            "Look-away gaze": "#f59e0b",
            "Unknown gaze": "#94a3b8"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.1f} s",
        textposition="outside"
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        title_font_color="#e2e8f0",
        xaxis_title="Gaze state",
        yaxis_title="Seconds",
        showlegend=False,
        margin=dict(l=10, r=10, t=55, b=10)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

closed_col, info_col = st.columns([1, 3])

with closed_col:
    st.metric(
        "😴 Eyes closed (sec)",
        "{:.2f}".format(closed_time)
    )

with info_col:
    st.markdown(
        '<div class="dashboard-note">'
        'These values describe signals recorded during the selected '
        'session. They are not a medical diagnosis.'
        '</div>',
        unsafe_allow_html=True
    )


# --------------------------------------------------
# EXPLORATORY SIGNAL ANALYSIS
# --------------------------------------------------

st.divider()
st.subheader("🧠 Exploratory Signal Analysis")

st.caption(
    "This section reads session_logs/overload_analysis.csv. "
    "It summarizes camera signals; it does not diagnose cognitive overload."
)

analysis = load_csv(ANALYSIS_FILE)

if not analysis.empty and "Session_ID" in analysis.columns:
    analysis["Session_ID"] = (
        analysis["Session_ID"].astype(str)
    )

if analysis.empty:
    st.info(
        "Analysis report not found. Run "
        "`python overload_analysis.py` to create it."
    )

elif "Session_ID" not in analysis.columns:
    st.warning(
        "Session_ID column is missing from overload_analysis.csv."
    )

else:
    selected_analysis = analysis[
        analysis["Session_ID"] == selected_session
    ].copy()

    if selected_analysis.empty:
        st.info(
            "No signal-analysis row exists for the selected session."
        )

    else:
        row = selected_analysis.iloc[-1]

        def analysis_number(column):
            value = pd.to_numeric(
                pd.Series([row.get(column, 0)]),
                errors="coerce"
            ).iloc[0]

            return (
                0.0 if pd.isna(value)
                else float(value)
            )

        a1, a2, a3, a4 = st.columns(4)

        with a1:
            st.metric(
                "↗️ Look-away / minute",
                "{:.2f}".format(
                    analysis_number("Look_Away_Per_Minute")
                )
            )

        with a2:
            st.metric(
                "👁️ Blinks / minute",
                "{:.2f}".format(
                    analysis_number("Blink_Per_Minute")
                )
            )

        with a3:
            st.metric(
                "❔ Unknown gaze",
                "{:.1f}%".format(
                    analysis_number("Unknown_Gaze_Percent")
                )
            )

        with a4:
            st.metric(
                "😴 Eyes closed",
                "{:.1f}%".format(
                    analysis_number("Eyes_Closed_Percent")
                )
            )

        st.markdown(
            "**Data quality:** " +
            str(row.get("Data_Quality", "Not specified"))
        )

        st.markdown(
            "**Exploratory notes:** " +
            str(row.get("Signal_Notes", "No notes recorded"))
        )

        percentage_columns = [
            ("Look-away time", "Look_Away_Percent"),
            ("Unknown-gaze time", "Unknown_Gaze_Percent"),
            ("Eyes-closed time", "Eyes_Closed_Percent")
        ]

        chart_rows = []

        for label, column in percentage_columns:
            if column in selected_analysis.columns:
                chart_rows.append({
                    "Signal": label,
                    "Percentage": analysis_number(column)
                })

        if chart_rows:
            signal_chart = px.bar(
                pd.DataFrame(chart_rows),
                x="Signal",
                y="Percentage",
                text="Percentage",
                title="Selected Session — Signal Percentages",
                color="Signal",
                color_discrete_map={
                    "Look-away time": "#f59e0b",
                    "Unknown-gaze time": "#94a3b8",
                    "Eyes-closed time": "#a78bfa"
                }
            )

            signal_chart.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            signal_chart.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                title_font_color="#e2e8f0",
                yaxis_title="Percent of session",
                xaxis_title="",
                showlegend=False,
                yaxis_range=[0, 100],
                margin=dict(l=10, r=10, t=55, b=10)
            )

            st.plotly_chart(
                signal_chart,
                use_container_width=True
            )


        # --------------------------------------------------
        # SESSION TREND ANALYSIS
        # --------------------------------------------------

        st.divider()
        st.subheader("📈 Session Trend Analysis")

        trend = analysis.drop_duplicates(
            subset=["Session_ID"],
            keep="last"
        ).copy()

        trend = trend.sort_values("Session_ID")

        if "Data_Quality" in trend.columns:
            trend = trend[
                trend["Data_Quality"]
                == "Usable for exploratory analysis"
            ].copy()

        trend_columns = [
            ("Look-away time", "Look_Away_Percent"),
            ("Unknown-gaze time", "Unknown_Gaze_Percent"),
            ("Eyes-closed time", "Eyes_Closed_Percent")
        ]

        available_trend_columns = [
            (label, column)
            for label, column in trend_columns
            if column in trend.columns
        ]

        if len(trend) >= 2 and available_trend_columns:
            trend_parts = []

            for label, column in available_trend_columns:
                values = pd.to_numeric(
                    trend[column],
                    errors="coerce"
                )

                trend_parts.append(
                    pd.DataFrame({
                        "Session_ID": trend["Session_ID"].values,
                        "Signal": label,
                        "Percentage": values.values
                    })
                )

            trend_long = pd.concat(
                trend_parts,
                ignore_index=True
            ).dropna(subset=["Percentage"])

            trend_chart = px.line(
                trend_long,
                x="Session_ID",
                y="Percentage",
                color="Signal",
                markers=True,
                title="Signal Percentages Across Sessions",
                color_discrete_map={
                    "Look-away time": "#f59e0b",
                    "Unknown-gaze time": "#94a3b8",
                    "Eyes-closed time": "#a78bfa"
                }
            )

            trend_chart.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                title_font_color="#e2e8f0",
                xaxis_title="Session ID (chronological)",
                yaxis_title="Percent of session",
                yaxis_ticksuffix="%",
                margin=dict(l=10, r=10, t=55, b=10)
            )

            st.plotly_chart(
                trend_chart,
                use_container_width=True
            )

            if "Blink_Per_Minute" in trend.columns:
                trend["Blink_Per_Minute"] = pd.to_numeric(
                    trend["Blink_Per_Minute"],
                    errors="coerce"
                )

                blink_trend = trend.dropna(
                    subset=["Blink_Per_Minute"]
                )

                if len(blink_trend) >= 2:
                    blink_chart = px.line(
                        blink_trend,
                        x="Session_ID",
                        y="Blink_Per_Minute",
                        markers=True,
                        title="Blink Rate Across Sessions"
                    )

                    blink_chart.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#e2e8f0",
                        title_font_color="#e2e8f0",
                        xaxis_title="Session ID",
                        yaxis_title="Blinks per minute"
                    )

                    st.plotly_chart(
                        blink_chart,
                        use_container_width=True
                    )

        else:
            st.info(
                "Record at least two usable sessions "
                "to display session trends."
            )

        st.markdown("**All session signal-analysis results**")

        display_columns = [
            "Session_ID",
            "duration_seconds",
            "Look_Away_Per_Minute",
            "Blink_Per_Minute",
            "Look_Away_Percent",
            "Unknown_Gaze_Percent",
            "Eyes_Closed_Percent",
            "Data_Quality",
            "Signal_Notes"
        ]

        display_columns = [
            column
            for column in display_columns
            if column in analysis.columns
        ]

        st.dataframe(
            analysis[display_columns].iloc[::-1],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Signal Analysis",
            data=analysis.to_csv(index=False).encode("utf-8"),
            file_name="overload_analysis_export.csv",
            mime="text/csv",
            use_container_width=True
        )

# --------------------------------------------------
# SIGNAL REVIEW
# --------------------------------------------------

st.divider()
st.subheader("🔎 Signal Review")

st.caption(
    "Exploratory signal flags based on configurable thresholds. "
    "These flags are not a validated cognitive overload prediction."
)

LOOK_AWAY_REVIEW = 30.0
UNKNOWN_GAZE_REVIEW = 30.0
EYES_CLOSED_REVIEW = 10.0
MIN_DURATION = 30.0


def review_number(row, column):
    value = pd.to_numeric(
        pd.Series([row.get(column)]),
        errors="coerce"
    ).iloc[0]

    return 0.0 if pd.isna(value) else float(value)


if analysis.empty or "Session_ID" not in analysis.columns:
    st.info(
        "Signal analysis is unavailable. Run "
        "`python overload_analysis.py` first."
    )

else:
    review_data = analysis.copy()
    review_data["Session_ID"] = (
        review_data["Session_ID"].astype(str)
    )

    selected_review = review_data[
        review_data["Session_ID"] == selected_session
    ]

    if selected_review.empty:
        st.info("No signal review data for this session.")

    else:
        row = selected_review.iloc[-1]

        duration_review = review_number(
            row, "duration_seconds"
        )

        lookaway_percent = review_number(
            row, "Look_Away_Percent"
        )

        unknown_percent = review_number(
            row, "Unknown_Gaze_Percent"
        )

        eyes_closed_percent = review_number(
            row, "Eyes_Closed_Percent"
        )

        flags = []

        if duration_review < MIN_DURATION:
            flags.append("Insufficient session duration")
        else:
            if lookaway_percent >= LOOK_AWAY_REVIEW:
                flags.append(
                    "Look-away percentage reached review threshold"
                )

            if unknown_percent >= UNKNOWN_GAZE_REVIEW:
                flags.append(
                    "Unknown-gaze percentage reached review threshold"
                )

            if eyes_closed_percent >= EYES_CLOSED_REVIEW:
                flags.append(
                    "Eyes-closed percentage reached review threshold"
                )

        st.markdown("**Selected session result**")

        if duration_review < MIN_DURATION:
            st.warning(
                "Insufficient data — record a longer session "
                "before interpreting these signals."
            )
        elif flags:
            st.warning(
                "Some signals need review. This does not mean "
                "cognitive overload is present."
            )
        else:
            st.success(
                "No configured review thresholds exceeded."
            )

        for flag in flags:
            st.write("- " + flag)

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Look-away",
            "{:.1f}%".format(lookaway_percent)
        )

        m2.metric(
            "Unknown gaze",
            "{:.1f}%".format(unknown_percent)
        )

        m3.metric(
            "Eyes closed",
            "{:.1f}%".format(eyes_closed_percent)
        )

        st.markdown("**Review results for all sessions**")

        review_rows = []

        for _, item in review_data.iterrows():
            item_duration = review_number(
                item, "duration_seconds"
            )

            item_lookaway = review_number(
                item, "Look_Away_Percent"
            )

            item_unknown = review_number(
                item, "Unknown_Gaze_Percent"
            )

            item_closed = review_number(
                item, "Eyes_Closed_Percent"
            )

            item_flags = []

            if item_duration < MIN_DURATION:
                result = "Insufficient data"
                item_flags.append("Short session")
            else:
                if item_lookaway >= LOOK_AWAY_REVIEW:
                    item_flags.append("Look-away review")

                if item_unknown >= UNKNOWN_GAZE_REVIEW:
                    item_flags.append("Gaze quality review")

                if item_closed >= EYES_CLOSED_REVIEW:
                    item_flags.append("Eye-closure review")

                result = (
                    "Signals need review"
                    if item_flags
                    else "No configured flags"
                )

            review_rows.append({
                "Session_ID": str(item["Session_ID"]),
                "Duration (sec)": round(item_duration, 2),
                "Look-away (%)": round(item_lookaway, 2),
                "Unknown gaze (%)": round(item_unknown, 2),
                "Eyes closed (%)": round(item_closed, 2),
                "Review result": result,
                "Flags": "; ".join(item_flags)
                if item_flags else "None"
            })

        if review_rows:
            review_table = pd.DataFrame(review_rows)

            st.dataframe(
                review_table,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Total sessions reviewed: {}".format(
                    len(review_table)
                )
            )


# --------------------------------------------------
# PERSONALIZED BASELINE
# --------------------------------------------------

st.divider()
st.subheader("🎯 Personalized Baseline")

st.caption(
    "A prototype baseline compares the selected session with the "
    "median of other usable sessions. This assumes the sessions "
    "belong to the same person."
)

if analysis.empty or "Session_ID" not in analysis.columns:
    st.info(
        "Run `python overload_analysis.py` first."
    )

else:
    baseline_data = analysis.copy()

    baseline_data["Session_ID"] = (
        baseline_data["Session_ID"].astype(str)
    )

    if "duration_seconds" in baseline_data.columns:
        baseline_data["duration_seconds"] = pd.to_numeric(
            baseline_data["duration_seconds"],
            errors="coerce"
        )

        baseline_data = baseline_data[
            baseline_data["duration_seconds"] >= 30
        ].copy()

    else:
        baseline_data = pd.DataFrame()

    if (
        "Data_Quality" in baseline_data.columns
        and not baseline_data.empty
    ):
        baseline_data = baseline_data[
            baseline_data["Data_Quality"]
            == "Usable for exploratory analysis"
        ].copy()

    baseline_metrics = [
        ("Look-away / minute", "Look_Away_Per_Minute", "events/min"),
        ("Blink rate", "Blink_Per_Minute", "blinks/min"),
        ("Look-away time", "Look_Away_Percent", "%"),
        ("Unknown gaze", "Unknown_Gaze_Percent", "%"),
        ("Eyes closed", "Eyes_Closed_Percent", "%")
    ]

    other_sessions = baseline_data[
        baseline_data["Session_ID"] != selected_session
    ].copy()

    if len(other_sessions) >= 2:
        baseline_source = other_sessions

        baseline_note = (
            "Baseline uses other usable sessions, excluding "
            "the selected session."
        )

    elif len(baseline_data) >= 2:
        baseline_source = baseline_data

        baseline_note = (
            "Only one other usable session is available. "
            "The baseline includes the selected session."
        )

    else:
        baseline_source = pd.DataFrame()
        baseline_note = ""

    if baseline_source.empty:
        st.info(
            "Record at least two usable sessions "
            "(30 seconds or longer) for a baseline."
        )

    else:
        st.markdown(
            "**Baseline method:** median of usable sessions."
        )

        st.write(baseline_note)

        selected_rows = analysis[
            analysis["Session_ID"].astype(str) == selected_session
        ]

        if selected_rows.empty:
            st.info("No analysis row for the selected session.")

        else:
            selected_row = selected_rows.iloc[-1]
            baseline_rows = []

            for label, column, unit in baseline_metrics:
                if column not in baseline_source.columns:
                    continue

                baseline_values = pd.to_numeric(
                    baseline_source[column],
                    errors="coerce"
                ).dropna()

                current_value = pd.to_numeric(
                    pd.Series([selected_row.get(column)]),
                    errors="coerce"
                ).iloc[0]

                if (
                    baseline_values.empty
                    or pd.isna(current_value)
                ):
                    continue

                median_value = float(
                    baseline_values.median()
                )

                baseline_rows.append({
                    "Metric": label,
                    "Selected session": round(
                        float(current_value), 2
                    ),
                    "Baseline median": round(
                        median_value, 2
                    ),
                    "Difference": round(
                        float(current_value) - median_value, 2
                    ),
                    "Unit": unit
                })

            if baseline_rows:
                baseline_table = pd.DataFrame(
                    baseline_rows
                )

                st.dataframe(
                    baseline_table,
                    use_container_width=True,
                    hide_index=True
                )

                plot_rows = []

                for item in baseline_rows:
                    plot_rows.append({
                        "Metric": item["Metric"],
                        "Measurement": "Selected session",
                        "Value": item["Selected session"]
                    })

                    plot_rows.append({
                        "Metric": item["Metric"],
                        "Measurement": "Baseline median",
                        "Value": item["Baseline median"]
                    })

                baseline_chart = px.bar(
                    pd.DataFrame(plot_rows),
                    x="Metric",
                    y="Value",
                    color="Measurement",
                    barmode="group",
                    title="Selected Session vs Baseline Median",
                    color_discrete_map={
                        "Selected session": "#38bdf8",
                        "Baseline median": "#f59e0b"
                    }
                )

                baseline_chart.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0",
                    title_font_color="#e2e8f0",
                    margin=dict(l=10, r=10, t=55, b=10)
                )

                st.plotly_chart(
                    baseline_chart,
                    use_container_width=True
                )

                st.download_button(
                    "Download Baseline Comparison",
                    data=baseline_table.to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name="personalized_baseline_comparison.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            st.markdown(
                '<div class="dashboard-note">'
                'This is a descriptive prototype, not a validated overload '
                'detector. Signal changes may have many causes.'
                '</div>',
                unsafe_allow_html=True
            )


# --------------------------------------------------
# LIVE BROWSER WEBCAM
# --------------------------------------------------

from live_camera import render_live_camera

render_live_camera()


# --------------------------------------------------

# --------------------------------------------------
# SESSION HISTORY
# --------------------------------------------------

st.divider()
st.subheader("📚 Session History")

history_columns = [
    "Session_ID",
    "session_start",
    "duration_seconds",
    "blink_count",
    "look_away_count",
    "eye_closure_warning_count",
    "center_gaze_seconds",
    "look_away_seconds",
    "unknown_gaze_seconds"
]

available_columns = [
    column for column in history_columns
    if column in summary.columns
]

if available_columns:
    history = (
        summary[available_columns]
        .iloc[::-1]
        .reset_index(drop=True)
        .copy()
    )

    # Recalculate counts from the loaded event log
    if (
                                "events" in globals()
                        and isinstance(events, pd.DataFrame)
                    and not events.empty
                and {"Session_ID", "Event"}.issubset(events.columns)
            and "Session_ID" in history.columns
    ):
        event_data = events.drop_duplicates().copy()

        event_data["Session_ID"] = (
            event_data["Session_ID"].astype(str).str.strip()
        )
        history["Session_ID"] = (
            history["Session_ID"].astype(str).str.strip()
        )
        event_data["Event"] = (
            event_data["Event"].astype(str).str.strip().str.upper()
        )

        # Count event types
        blink_counts = (
            event_data[event_data["Event"] == "BLINK"]
                .groupby("Session_ID")
                .size()
        )

        lookaway_counts = (
            event_data[event_data["Event"] == "LOOK_AWAY_WARNING"]
                .groupby("Session_ID")
                .size()
        )

        eye_warning_names = [
            "LONG_EYE_CLOSURE_WARNING",
            "EYE_CLOSURE_WARNING",
            "EYES_CLOSED_WARNING",
            "EYE_WARNING",
        ]

        eye_warning_counts = (
            event_data[event_data["Event"].isin(eye_warning_names)]
                .groupby("Session_ID")
                .size()
        )

        # Update counts for each session
        for idx, session_id in history["Session_ID"].items():
            session_events = event_data[
                event_data["Session_ID"] == session_id
                ]

            if not session_events.empty:
                if "blink_count" in history.columns:
                    history.at[idx, "blink_count"] = int(
                        blink_counts.get(session_id, 0)
                    )

                if "look_away_count" in history.columns:
                    history.at[idx, "look_away_count"] = int(
                        lookaway_counts.get(session_id, 0)
                    )

                if "eye_closure_warning_count" in history.columns:
                    history.at[idx, "eye_closure_warning_count"] = int(
                        eye_warning_counts.get(session_id, 0)
                    )
    history.rename(columns={
        "Session_ID": "Session ID",
        "session_start": "Start time",
        "duration_seconds": "Duration (s)",
        "blink_count": "Blinks",
        "look_away_count": "Look-away",
        "eye_closure_warning_count": "Eye warnings",
        "center_gaze_seconds": "Center gaze (s)",
        "look_away_seconds": "Look-away (s)",
        "unknown_gaze_seconds": "Unknown gaze (s)"
    }, inplace=True)

    for column in [
        "Duration (s)",
        "Center gaze (s)",
        "Look-away (s)",
        "Unknown gaze (s)"
    ]:
        if column in history.columns:
            history[column] = pd.to_numeric(
                history[column], errors="coerce"
            ).round(2)

    st.dataframe(
        history,
        use_container_width=True,
        hide_index=True
    )

else:
    st.info("No session history columns are available.")



# --------------------------------------------------
# SESSION COMPARISON
# --------------------------------------------------

st.divider()
st.header("📊 Session Comparison")

if analysis.empty:
    st.info("No session analysis data available.")

else:
    compare_df = analysis.copy()

    numeric_columns = [
        "Look_Away_Percent",
        "Unknown_Gaze_Percent",
        "Eyes_Closed_Percent",
        "Blink_Per_Minute",
        "Look_Away_Per_Minute"
    ]

    for column in numeric_columns:
        if column in compare_df.columns:
            compare_df[column] = pd.to_numeric(
                compare_df[column],
                errors="coerce"
            )

    if "Data_Quality" in compare_df.columns:
        compare_df = compare_df[
            compare_df["Data_Quality"]
            == "Usable for exploratory analysis"
        ].copy()

    elif "duration_seconds" in compare_df.columns:
        compare_df = compare_df[
            pd.to_numeric(
                compare_df["duration_seconds"],
                errors="coerce"
            ) >= 30
        ].copy()

    if compare_df.empty:
        st.warning(
            "No usable sessions available for comparison."
        )

    else:
        metric_labels = {
            "Look_Away_Percent": "Look-away (%)",
            "Unknown_Gaze_Percent": "Unknown gaze (%)",
            "Eyes_Closed_Percent": "Eyes closed (%)",
            "Blink_Per_Minute": "Blink rate (/min)",
            "Look_Away_Per_Minute": "Look-away rate (/min)"
        }

        available_metrics = [
            column
            for column in metric_labels
            if column in compare_df.columns
        ]

        selected_metrics = st.multiselect(
            "Choose metrics",
            options=available_metrics,
            default=[
                column
                for column in [
                    "Look_Away_Percent",
                    "Unknown_Gaze_Percent",
                    "Eyes_Closed_Percent"
                ]
                if column in available_metrics
            ],
            format_func=lambda x: metric_labels.get(x, x)
        )

        if selected_metrics:
            chart_df = compare_df.set_index(
                "Session_ID"
            )[selected_metrics]

            chart_df = chart_df.rename(
                columns=metric_labels
            )

            st.line_chart(
                chart_df,
                height=350
            )

            st.dataframe(
                chart_df,
                use_container_width=True
            )

        else:
            st.info(
                "Select at least one metric to display the chart."
            )


# --------------------------------------------------
# COMPARE GAZE TIME ACROSS SESSIONS
# --------------------------------------------------

st.divider()
st.subheader("📈 Compare Gaze Time")

comparison_columns = [
    "Session_ID",
    "center_gaze_seconds",
    "look_away_seconds",
    "unknown_gaze_seconds"
]

comparison_columns = [
    column
    for column in comparison_columns
    if column in summary.columns
]

if len(summary) >= 2 and len(comparison_columns) > 1:
    comparison = summary[
        comparison_columns
    ].copy().melt(
        id_vars=["Session_ID"],
        var_name="Metric",
        value_name="Seconds"
    )

    comparison["Metric"] = comparison["Metric"].map({
        "center_gaze_seconds": "Center gaze",
        "look_away_seconds": "Look-away gaze",
        "unknown_gaze_seconds": "Unknown gaze"
    }).fillna(comparison["Metric"])

    fig = px.bar(
        comparison,
        x="Session_ID",
        y="Seconds",
        color="Metric",
        barmode="group",
        title="Gaze Time Across Sessions",
        color_discrete_map={
            "Center gaze": "#38bdf8",
            "Look-away gaze": "#f59e0b",
            "Unknown gaze": "#94a3b8"
        }
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        title_font_color="#e2e8f0",
        xaxis_title="Session ID",
        yaxis_title="Seconds"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:
    st.info(
        "Complete at least two sessions to compare history."
    )


# --------------------------------------------------
# EVENT LOG
# --------------------------------------------------

st.divider()
st.subheader("🧾 Event Log")
st.caption("Events recorded for the selected session")

if not session_events.empty:
    event_display = session_events.drop_duplicates().copy()

    if "Elapsed_seconds" in event_display.columns:
        event_display["Elapsed_seconds"] = pd.to_numeric(
            event_display["Elapsed_seconds"],
            errors="coerce"
        ).round(2)

    st.dataframe(
        event_display.iloc[::-1],
        use_container_width=True,
        hide_index=True
    )

else:
    st.info(
        "No event records found for this session."
    )


# --------------------------------------------------
# EXPORT REPORTS
# --------------------------------------------------

st.divider()
st.subheader("⬇️ Export Reports")

d1, d2 = st.columns(2)

with d1:
    st.download_button(
        "Download Session History",
        data=summary.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="session_summary_export.csv",
        mime="text/csv",
        use_container_width=True
    )

with d2:
    if not events.empty:
        st.download_button(
            "Download Event Log",
            data=events.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="event_log_export.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info(
            "No event log data available to export."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Neuro-Adaptive Interface • Analytics dashboard. "
    "This dashboard reports observed camera signals. "
    "Gaze and blink metrics do not independently diagnose "
    "cognitive overload or a medical condition."
)