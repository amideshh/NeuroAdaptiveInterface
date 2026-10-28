import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(
    BASE_DIR, "session_logs", "session_summary.csv"
)

df = pd.read_csv(CSV_FILE)

required_columns = [
    "Session_ID",
    "duration_seconds",
    "blink_count",
    "look_away_count",
    "eye_closure_warning_count",
    "center_gaze_seconds",
    "look_away_seconds",
    "unknown_gaze_seconds",
    "eyes_closed_seconds",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    print("Missing columns:", missing)
    raise SystemExit(1)

numeric_columns = required_columns[1:]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# Avoid unreliable estimates from very short sessions.
df["Data_Quality"] = "Usable for exploratory analysis"

df.loc[
    (df["duration_seconds"] < 30) |
    (df["duration_seconds"] <= 0),
    "Data_Quality"
] = "Insufficient session duration"

# Calculate normalized measures.
duration = df["duration_seconds"].where(
    df["duration_seconds"] > 0
)

df["Look_Away_Per_Minute"] = (
    df["look_away_count"] / duration * 60
).fillna(0).round(2)

df["Blink_Per_Minute"] = (
    df["blink_count"] / duration * 60
).fillna(0).round(2)

df["Look_Away_Percent"] = (
    df["look_away_seconds"] / duration * 100
).fillna(0).round(2)

df["Unknown_Gaze_Percent"] = (
    df["unknown_gaze_seconds"] / duration * 100
).fillna(0).round(2)

df["Eyes_Closed_Percent"] = (
    df["eyes_closed_seconds"] / duration * 100
).fillna(0).round(2)

# This is an exploratory signal summary, NOT a diagnosis.
df["Signal_Notes"] = ""

df.loc[
    df["Look_Away_Percent"] >= 30,
    "Signal_Notes"
] += "High look-away time; "

df.loc[
    df["Unknown_Gaze_Percent"] >= 30,
    "Signal_Notes"
] += "High unknown-gaze time; "

df.loc[
    df["eye_closure_warning_count"] > 0,
    "Signal_Notes"
] += "Eye-closure warning recorded; "

df.loc[
    df["Signal_Notes"] == "",
    "Signal_Notes"
] = "No configured signal flag"

output_file = os.path.join(
    BASE_DIR, "session_logs", "overload_analysis.csv"
)

df.to_csv(output_file, index=False)

columns_to_show = [
    "Session_ID",
    "duration_seconds",
    "Look_Away_Per_Minute",
    "Blink_Per_Minute",
    "Look_Away_Percent",
    "Unknown_Gaze_Percent",
    "Eyes_Closed_Percent",
    "Data_Quality",
    "Signal_Notes",
]

print("\nSESSION SIGNAL ANALYSIS")
print(df[columns_to_show].to_string(index=False))

print("\nSaved report to:")
print(output_file)

print(
    "\nNote: These are exploratory camera-signal indicators. "
    "They do not confirm cognitive overload or diagnose a medical condition."
)