import pandas as pd
from prepare_data import load_clean_data

TARGET_DISTANCE_M = 5000


def build_prediction_dataset(df):
    """
    One row per 5k race. Features come from the athlete's most recent
    *prior* race (any distance), so nothing after the target race leaks in.
    """
    rows = []

    for athlete_id, group in df.groupby("athlete_id"):
        group = group.sort_values("race_date")
        fives = group[group["event_name"] == "5000m"]

        for _, race in fives.iterrows():
            prior = group[group["race_date"] < race["race_date"]]
            if prior.empty:
                continue
            last = prior.iloc[-1]

            prior_5k_equivs = riegel_predict(prior["time_seconds"], prior["distance_m"])

            rows.append({
                "athlete_id": athlete_id,
                "target_date": race["race_date"],
                "target_time_seconds": race["time_seconds"],
                "prior_time_seconds": last["time_seconds"],
                "prior_distance_m": last["distance_m"],
                "days_since_prior": (race["race_date"] - last["race_date"]).days,
                "gender": race.get("gender"),
                "race_count_so_far": len(prior),
                "prior_is_5k": int(last["distance_m"] == 5000),
                "best_prior_5k_equiv": prior_5k_equivs.min(),
                "altitude": race.get("altitude"),
            })

    return pd.DataFrame(rows)


def riegel_predict(t1, d1, d2=TARGET_DISTANCE_M, exponent=1.06):
    return t1 * (d2 / d1) ** exponent


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
