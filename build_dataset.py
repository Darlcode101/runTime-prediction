import pandas as pd
from prepare_data import load_clean_data
from features import TARGET_DISTANCE_M, compute_features, riegel_predict


def build_prediction_dataset(df):
    """
    One row per 5k race. Features come from the athlete's races *prior* to
    that race, so nothing after the target race leaks in. Uses the same
    compute_features() that predict.py uses at inference time, so training
    and serving can't drift apart.
    """
    rows = []

    for athlete_id, group in df.groupby("athlete_id"):
        group = group.sort_values("race_date")
        fives = group[group["event_name"] == "5000m"]

        for _, race in fives.iterrows():
            prior = group[group["race_date"] < race["race_date"]]
            if prior.empty:
                continue

            prior_races = [
                {"date": r["race_date"], "distance_m": r["distance_m"], "time_seconds": r["time_seconds"]}
                for _, r in prior.iterrows()
            ]
            feats = compute_features(
                prior_races,
                target_date=race["race_date"],
                gender=race.get("gender"),
                altitude=race.get("altitude"),
                temperature=race.get("temperature"),
                wind_speed=race.get("wind_speed"),
                humidity=race.get("humidity"),
            )

            rows.append({
                "athlete_id": athlete_id,
                "target_date": race["race_date"],
                "target_time_seconds": race["time_seconds"],
                "gender": race.get("gender"),
                **feats,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_clean_data()
    dataset = build_prediction_dataset(df)
    print("Predictable 5k races (has a prior race):", len(dataset))
    print(dataset.head())

    dataset["riegel_pred"] = riegel_predict(
        dataset["prior_time_seconds"], dataset["prior_distance_m"]
    )
    dataset["riegel_error"] = dataset["riegel_pred"] - dataset["target_time_seconds"]
    print("\nRiegel baseline MAE (seconds):", dataset["riegel_error"].abs().mean())

    dataset.to_csv("prediction_dataset.csv", index=False)
    print("\nSaved to prediction_dataset.csv")
