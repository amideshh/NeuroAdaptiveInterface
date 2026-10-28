
import csv
import os

file_path = os.path.join(
    "session_logs",
    "overload_analysis.csv"
)

with open(file_path, "r", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

print("Total sessions:", len(rows))
print("Available columns:")
print(reader.fieldnames if reader.fieldnames else "No columns")

print("\nSession Data Check")
print("-" * 40)

for row in rows:
    print("Session:", row["Session_ID"])
    print("Duration:", row["duration_seconds"])
    print("Look Away %:", row["Look_Away_Percent"])
    print("Unknown Gaze %:", row["Unknown_Gaze_Percent"])
    print("Eyes Closed %:", row["Eyes_Closed_Percent"])
    print("Data Quality:", row["Data_Quality"])
    print("-" * 40)

print("\nActual overload labels:",
      "Not available in this CSV")
