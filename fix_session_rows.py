
import os
import shutil
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(
    BASE_DIR, "session_logs", "session_summary.csv"
)

BACKUP_FILE = os.path.join(
    BASE_DIR, "session_logs",
    "session_summary_backup_before_fix.csv"
)

if not os.path.exists(BACKUP_FILE):
    raise FileNotFoundError("Backup file not found. Stopping.")

df = pd.read_csv(CSV_FILE)

fix_values = {
    "20261027_130558": {
        "blink_count": 9,
        "look_away_count": 20,
        "center_gaze_seconds": 50.99,
        "look_away_seconds": 16.66,
        "unknown_gaze_seconds": 21.88,
    },
    "20261027_184101": {
        "blink_count": 2,
        "look_away_count": 0,
        "center_gaze_seconds": 10.22,
        "look_away_seconds": 0.00,
        "unknown_gaze_seconds": 6.32,
    },
}

for session_id, values in fix_values.items():
    mask = df["Session_ID"].astype(str) == session_id

    if mask.sum() != 1:
        raise ValueError(
            "{} must occur exactly once.".format(session_id)
        )

    for column, value in values.items():
        if pd.isna(df.loc[mask, column]).all():
            df.loc[mask, column] = value

    # Copy verified values into corresponding legacy columns.
    aliases = {
        "look_away_count": "Look_Away_Count",
        "blink_count": "Blink_Count",
        "look_away_seconds": "Look_Away_Seconds",
        "center_gaze_seconds": "Center_Seconds",
        "unknown_gaze_seconds": "Unknown_Gaze_Seconds",
    }

    for source, target in aliases.items():
        if pd.isna(df.loc[mask, source]).all():
            df.loc[mask, source] = df.loc[mask, target].values

# Preserve the original CSV before writing the corrected version.
shutil.copy2(
    CSV_FILE,
    CSV_FILE + ".pre_fix_backup"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "session_logs",
    "session_summary_fixed.csv"
)

df.to_csv(OUTPUT_FILE, index=False)

print("Fixed data saved to:", OUTPUT_FILE)

print("Selected session rows updated.")
print(
    df[df["Session_ID"].astype(str).isin(fix_values)]
    .to_string(index=False)
)
print("Additional backup:", CSV_FILE + ".pre_fix_backup")
