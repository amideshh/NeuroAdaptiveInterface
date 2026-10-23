
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "session_logs")

summary_path = os.path.join(LOG_DIR, "session_summary.csv")
events_path = os.path.join(LOG_DIR, "event_log.csv")

summary = pd.read_csv(summary_path)
events = pd.read_csv(events_path)

print("\n===== SESSION DATA VALIDATION =====")

for _, session in summary.iterrows():
    session_id = session["Session_ID"]
    duration = float(session["duration_seconds"])

    session_events = events[
        events["Session_ID"] == session_id
    ]

    # Count blink events
    blink_events = (
        session_events["Event"] == "BLINK"
    ).sum()

    # Count eye-closure warning events
    eye_warnings = (
        session_events["Event"] == "LONG_EYE_CLOSURE_WARNING"
    ).sum()

    # Check gaze-time total
    gaze_total = (
        float(session["center_gaze_seconds"])
        + float(session["look_away_seconds"])
        + float(session["unknown_gaze_seconds"])
    )

    print("\nSession:", session_id)
    print("Duration:", round(duration, 2), "seconds")
    print("Gaze total:", round(gaze_total, 2), "seconds")
    print("Blink summary:", int(session["blink_count"]))
    print("Blink events:", int(blink_events))
    print("Eye warnings summary:",
          int(session["eye_closure_warning_count"]))
    print("Eye warning events:", int(eye_warnings))

    if abs(duration - gaze_total) <= 0.15:
        print("Gaze duration: PASS")
    else:
        print("Gaze duration: CHECK")

    if int(session["blink_count"]) == int(blink_events):
        print("Blink count: PASS")
    else:
        print("Blink count: CHECK")

    if int(session["eye_closure_warning_count"]) == int(eye_warnings):
        print("Eye warning count: PASS")
    else:
        print("Eye warning count: CHECK")

print("\n===== VALIDATION COMPLETE =====")
