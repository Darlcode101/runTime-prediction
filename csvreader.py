import pandas as pd

df = pd.read_csv("joined.csv")

# keep athletes with at least 3 races
race_counts = df.groupby("athlete_id").size()

valid_athletes = race_counts[race_counts >= 3].index

df = df[df["athlete_id"].isin(valid_athletes)]

# require each athlete to have at least one 5k
athletes_with_5k = df[
    df["event_name"].str.lower().isin(["5000m", "5k", "5km"])
]["athlete_id"].unique()

df = df[df["athlete_id"].isin(athletes_with_5k)]

print(df.shape)
print(df["athlete_id"].nunique())
print(df.head())