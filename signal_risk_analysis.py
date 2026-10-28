
import csv
import os

file_path = os.path.join(
    "session_logs",
    "overload_analysis.csv"
)

# These are exploratory review thresholds,
# NOT medically validated overload thresholds.
LOOK_AWAY_REVIEW = 30.0
UNKNOWN_GAZE_REVIEW = 30.0
EYES_CLOSED_REVIEW = 10.0
MIN_DURATION = 30.0

with open(file_path, "r", newline="") as file:
    rows = list(csv.DictReader(file))

print("\nSIGNAL REVIEW REPORT")
print("=" * 50)

for row in rows:
    duration = float(row["duration_seconds"])
    look_away = float(row["Look_Away_Percent"])
    unknown = float(row["Unknown_Gaze_Percent"])
    eyes_closed = float(row["Eyes_Closed_Percent"])

    flags = []

    if duration < MIN_DURATION:
        flags.append("Short session")
    else:
        if look_away >= LOOK_AWAY_REVIEW:
            flags.append("Review look-away time")

        if unknown >= UNKNOWN_GAZE_REVIEW:
            flags.append("Review gaze data quality")

        if eyes_closed >= EYES_CLOSED_REVIEW:
            flags.append("Review eye-closure signal")

    print("\nSession:", row["Session_ID"])

    if duration < MIN_DURATION:
        print("Result: Insufficient data")
    elif flags:
        print("Result: Signals need review")
    else:
        print("Result: No configured signal flags")

    for flag in flags:
        print("-", flag)

    if not flags and duration >= MIN_DURATION:
        print("- No configured review threshold exceeded")

print("\nNote: This is exploratory signal review,")
print("not a validated cognitive overload prediction.")
