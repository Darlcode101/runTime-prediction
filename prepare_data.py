import pandas as pd

EVENT_DISTANCES_M = {
    "1600m": 1600,
    "Mile": 1609.34,
    "3000m": 3000,
    "3200m": 3200,
    "2 Mile": 3218.68,
    "5000m": 5000,
    "4 Mile": 6437.38,
    "5 Mile": 8046.72,
    "6000m": 6000,
    "8000m": 8000,
    "10000m": 10000,
}


def time_to_seconds(time_str):
    parts = time_str.split(":")
    minutes = float(parts[0])
    seconds = float(parts[1])
    return minutes * 60 + seconds


def load_clean_data(path="joined.csv"):
    df = pd.read_csv(path, low_memory=False)

    df = df[df["event_name"].isin(EVENT_DISTANCES_M.keys())]
    df = df.dropna(subset=["result_time", "athlete_id"])

    df = df.drop_duplicates(
        subset=["athlete_id", "event_name", "result_time", "date_of_event"]
    )

    race_counts = df.groupby("athlete_id").size()
    valid_athletes = race_counts[race_counts >= 3].index
    df = df[df["athlete_id"].isin(valid_athletes)]

    athletes_with_5k = df[df["event_name"] == "5000m"]["athlete_id"].unique()
    df = df[df["athlete_id"].isin(athletes_with_5k)]

    df["time_seconds"] = df["result_time"].apply(time_to_seconds)
    df["distance_m"] = df["event_name"].map(EVENT_DISTANCES_M)

    df["race_date"] = df["date_of_event"].fillna(df["start_date"])
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df = df.dropna(subset=["race_date"])

    df = df.drop_duplicates(subset=["athlete_id", "event_name", "result_time"])
    df = df.sort_values(["athlete_id", "race_date"])

    return df


if __name__ == "__main__":
    df = load_clean_data()
    print("Athletes:", df["athlete_id"].nunique())
    print("Races:", len(df))
    print(df["event_name"].value_counts())
