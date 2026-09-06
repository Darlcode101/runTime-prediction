import pandas as pd

df = pd.read_csv("joined.csv")

valid_events = [
    "1600m",
    "Mile",
    "3000m",
    "3200m",
    "2 Mile",
    "5000m",
    "4 Mile",
    "5 Mile",
    "6000m",
    "8000m",
    "10000m"
]

# 1. keep only useful race distances
df = df[df["event_name"].isin(valid_events)]

# 2. remove duplicate race results
df = df.drop_duplicates(
    subset=[
        "athlete_id",
        "event_name",
        "result_time",
        "date_of_event"
    ]
)

# 3. keep athletes with at least 3 valid races
race_counts = df.groupby("athlete_id").size()

valid_athletes = race_counts[race_counts >= 3].index

df = df[df["athlete_id"].isin(valid_athletes)]

# 4. keep only athletes who have run at least one 5000m
athletes_with_5k = df[
    df["event_name"] == "5000m"
]["athlete_id"].unique()

df = df[df["athlete_id"].isin(athletes_with_5k)]

# 5. convert race times into seconds
def time_to_seconds(time):
    parts = time.split(":")

    minutes = float(parts[0])
    seconds = float(parts[1])

    return minutes * 60 + seconds

df["time_seconds"] = df["result_time"].apply(time_to_seconds)

# 6. sort by athlete and date
df = df.sort_values(
    ["athlete_id", "date_of_event"]
)

# 7. print some checks
print("Athletes:", df["athlete_id"].nunique())
print("Races:", len(df))

print("Missing dates:", df["date_of_event"].isna().sum())
print("Percent missing:", df["date_of_event"].isna().mean() * 100)

# remove exact duplicate performances for the same athlete/event/time
df = df.drop_duplicates(
    subset=[
        "athlete_id",
        "event_name",
        "result_time"
    ]
)
print("Athletes:", df["athlete_id"].nunique())
print("Races:", len(df))
missing_dates = df[df["date_of_event"].isna()]

print(
    missing_dates[
        ["date_of_event", "start_date", "event_name", "result_time"]
    ].head(20)
)
df["race_date"] = df["date_of_event"].fillna(df["start_date"])
df["race_date"] = pd.to_datetime(df["race_date"])
df = df.sort_values(["athlete_id", "race_date"])

print("Still missing dates:", df["race_date"].isna().sum())

print(
    df[
        [
            "athlete_id",
            "race_date",
            "event_name",
            "result_time",
            "time_seconds"
        ]
    ].head(50)
)
