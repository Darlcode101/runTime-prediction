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

# 1. keep only valid race distances
df = df[df["event_name"].isin(valid_events)]

# 2. count races per athlete
race_counts = df.groupby("athlete_id").size()

# 3. keep athletes with at least 3 valid races
valid_athletes = race_counts[race_counts >= 3].index

df = df[df["athlete_id"].isin(valid_athletes)]

print("Athletes:", df["athlete_id"].nunique())
print("Races:", len(df))

print(df[
    ["athlete_id", "event_name", "result_time"]
].head(30))