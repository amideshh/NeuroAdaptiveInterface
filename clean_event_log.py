
import os
import shutil
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "session_logs")

csv_path = os.path.join(LOG_DIR, "event_log.csv")
backup_path = os.path.join(LOG_DIR, "event_log_backup.csv")

if not os.path.exists(csv_path):
    print("ERROR: event_log.csv not found")

elif os.path.exists(backup_path):
    print("Backup already exists.")
    print("Original CSV was NOT changed.")
    print("Check the backup before proceeding.")

else:
    shutil.copy2(csv_path, backup_path)

    df = pd.read_csv(csv_path)
    before = len(df)

    cleaned = df.drop_duplicates()
    removed = before - len(cleaned)

    cleaned.to_csv(csv_path, index=False)

    print("Event log cleaning completed.")
    print("Rows before:", before)
    print("Duplicate rows removed:", removed)
    print("Rows after:", len(cleaned))
    print("Backup saved:", backup_path)
