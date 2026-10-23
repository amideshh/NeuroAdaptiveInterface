
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "session_logs")

summary = pd.read_csv(
    os.path.join(LOG_DIR, "session_summary.csv")
)

events = pd.read_csv(
    os.path.join(LOG_DIR, "event_log.csv")
)

print("\n===== LOOK-AWAY VALIDATION =====")

for _, session in summary.iterrows():
    session_id = session["Session_ID"]

    session_events = events[
        events["Session_ID"] == session_id
    ]

    warning_events = session_events[
        session_events["Event"] == "LOOK_AWAY_WARNING"
    ]

    summary_count = int(session["look_away_count"])
    event_count = len(warning_events)

    print("\nSession:", session_id)
    print("Summary look-away count:", summary_count)
    print("LOOK_AWAY_WARNING event count:", event_count)

    if summary_count == event_count:
        print("Counts match.")
    else:
        print("Counts differ; this may be expected.")
        print("Check the counter and warning logging logic.")

print("\n===== CHECK COMPLETE =====")

