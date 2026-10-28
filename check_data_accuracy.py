
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "session_logs")

files = {
    "Event Log": "event_log.csv",
    "Session Summary": "session_summary.csv",
    "Overload Analysis": "overload_analysis.csv",
}

for name, filename in files.items():
    path = os.path.join(LOG_DIR, filename)

    print("\n" + "=" * 50)
    print(name)

    if not os.path.exists(path):
        print("FILE NOT FOUND:", path)
        continue

    try:
        df = pd.read_csv(path)

        print("Rows:", len(df))
        print("Columns:", list(df.columns))
        print("Empty cells:", int(df.isna().sum().sum()))
        print("Duplicate rows:", int(df.duplicated().sum()))

        if df.empty:
            print("WARNING: File has no records.")
        else:
            print(df.tail(3).to_string(index=False))
            print("\nMissing values by column:")
            print(df.isna().sum()[df.isna().sum() > 0])
            print("\nMissing values by column:")
            print(df.isna().sum()[df.isna().sum() > 0])

            # 👇 Ee line-inu thaazhe new code paste cheyyuka
            if name in ["Session Summary", "Overload Analysis"]:
                print("\nMissing values by session:")

                missing_cols = df.columns[df.isna().any()].tolist()

                if "Session_ID" in df.columns:
                    print(
                        df.loc[
                            df[missing_cols].isna().any(axis=1),
                            ["Session_ID"] + [
                                col for col in missing_cols
                                if col != "Session_ID"
                            ]
                        ].to_string(index=False)
                    )

            if name == "Event Log":
                duplicate_mask = df.duplicated(keep=False)
                duplicate_rows = df[duplicate_mask].sort_values(
                    by=["Session_ID", "Timestamp", "Event"]
                )

                print("\nDuplicate rows found:", int(df.duplicated().sum()))
                print("\nExact duplicate groups:")
                print(duplicate_rows.to_string(index=False))

                print("\nMissing Details by event:")
                print(
                    df[df["Details"].isna()]
                        .groupby("Event")
                        .size()
                        .sort_values(ascending=False)
                )

    except Exception as e:
        print("ERROR:", e)

print("\nCSV CHECK COMPLETED")
