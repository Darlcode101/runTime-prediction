"""
Trains the model that actually gets deployed (predict.py / app.py use it).

Uses the feature set that won the ablation in train_models.py - history
features + weather - and trains on the *full* dataset (no train/test split)
since this is the artifact we ship, not the one we evaluate. Accuracy
numbers live in train_models.py's held-out comparison, not here.
"""

import json

import joblib
import pandas as pd
from xgboost import XGBRegressor

FEATURE_COLS = [
    "prior_time_seconds", "prior_distance_m", "days_since_prior", "gender_M",
    "race_count_so_far", "prior_is_5k", "best_prior_5k_equiv", "altitude",
    "temperature", "temperature_missing",
    "wind_speed", "wind_speed_missing",
    "humidity", "humidity_missing",
]

MODEL_PATH = "model/xgb_5k_model.joblib"
MEDIANS_PATH = "model/imputation_medians.json"

if __name__ == "__main__":
    import os
    os.makedirs("model", exist_ok=True)

    dataset = pd.read_csv("prediction_dataset.csv")

    dataset["gender"] = dataset["gender"].fillna("U")
    dataset["gender_M"] = (dataset["gender"] == "M").astype(int)

    medians = {}
    for col in ["altitude", "temperature", "wind_speed", "humidity"]:
        medians[col] = float(dataset[col].median())
        if col != "altitude":
            dataset[f"{col}_missing"] = dataset[col].isna().astype(int)
        dataset[col] = dataset[col].fillna(medians[col])

    X = dataset[FEATURE_COLS]
    y = dataset["target_time_seconds"]

    model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    with open(MEDIANS_PATH, "w") as f:
        json.dump(medians, f, indent=2)

    print(f"Trained on {len(X)} races. Saved model to {MODEL_PATH}")
    print(f"Saved imputation medians to {MEDIANS_PATH}: {medians}")
