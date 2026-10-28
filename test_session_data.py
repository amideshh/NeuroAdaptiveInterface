import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(
    BASE_DIR, "session_logs", "session_summary.csv"
)

df = pd.read_csv(CSV_FILE)

print("Total sessions:", len(df))
print("\nAvailable columns:")
print(df.columns.tolist())

print("\nLatest 5 sessions:")
print(df.tail(5).to_string(index=False))