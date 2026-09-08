"""
Predict a future 5k time from an athlete's recent race history.

CLI usage:
    python predict.py

Library usage:
    from predict import predict_5k
    predict_5k(
        races=[{"date": "2025-03-01", "distance_m": 3000, "time_seconds": 540}],
        gender="M",
    )
"""

import json
from datetime import date, datetime

import joblib
import pandas as pd

from features import compute_features, riegel_predict
from train_final_model import FEATURE_COLS, MEDIANS_PATH, MODEL_PATH

DISTANCE_CHOICES = {
    "1600m": 1600, "mile": 1609.34, "3000m": 3000, "3200m": 3200,
    "2 mile": 3218.68, "5000m": 5000, "4 mile": 6437.38,
    "5 mile": 8046.72, "6000m": 6000, "8000m": 8000, "10000m": 10000,
}

_model = None
_medians = None


def _load_artifacts():
    global _model, _medians
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        with open(MEDIANS_PATH) as f:
            _medians = json.load(f)
    return _model, _medians


def format_time(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:05.2f}"


def predict_5k(races, gender=None, target_date=None, altitude=None,
               temperature=None, wind_speed=None, humidity=None):
    """
    races: list of {"date": "YYYY-MM-DD" or date, "distance_m": float, "time_seconds": float}
           at least one race, most recent races matter most.
    Returns dict with the XGBoost prediction, the Riegel baseline for comparison,
    and the feature values that were used (useful for debugging/transparency).
    """
    if not races:
        raise ValueError("Need at least one prior race to predict from.")

    model, medians = _load_artifacts()

    target_date = target_date or date.today()
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()

    parsed_races = []
    for r in races:
        d = r["date"]
        if isinstance(d, str):
            d = datetime.strptime(d, "%Y-%m-%d").date()
        parsed_races.append({"date": d, "distance_m": float(r["distance_m"]), "time_seconds": float(r["time_seconds"])})

    feats = compute_features(
        parsed_races, target_date=target_date, gender=gender,
        altitude=altitude, temperature=temperature, wind_speed=wind_speed, humidity=humidity,
    )

    feats["altitude"] = feats["altitude"] if feats["altitude"] is not None else medians["altitude"]
    for col in ["temperature", "wind_speed", "humidity"]:
        feats[f"{col}_missing"] = int(feats[col] is None)
        feats[col] = feats[col] if feats[col] is not None else medians[col]

    X = pd.DataFrame([feats])[FEATURE_COLS]
    predicted_seconds = float(model.predict(X)[0])

    last = max(parsed_races, key=lambda r: r["date"])
    riegel_seconds = float(riegel_predict(last["time_seconds"], last["distance_m"]))

    return {
        "predicted_seconds": predicted_seconds,
        "predicted_time": format_time(predicted_seconds),
        "riegel_seconds": riegel_seconds,
        "riegel_time": format_time(riegel_seconds),
        "features_used": feats,
    }


def _prompt_races():
    print("Enter your recent races, most distances supported, oldest or newest order doesn't matter.")
    print(f"Distances: {', '.join(DISTANCE_CHOICES.keys())}")
    races = []
    while True:
        label = input("\nRace distance (blank to finish): ").strip().lower()
        if not label:
            break
        if label not in DISTANCE_CHOICES:
            print(f"  Unknown distance '{label}', pick one of: {', '.join(DISTANCE_CHOICES.keys())}")
            continue
        time_str = input("  Time (mm:ss, e.g. 18:45): ").strip()
        try:
            minutes, seconds = time_str.split(":")
            time_seconds = float(minutes) * 60 + float(seconds)
        except ValueError:
            print("  Couldn't parse that time, expected mm:ss.")
            continue
        date_str = input("  Date raced (YYYY-MM-DD): ").strip()
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            print("  Couldn't parse that date, expected YYYY-MM-DD.")
            continue
        races.append({"date": date_str, "distance_m": DISTANCE_CHOICES[label], "time_seconds": time_seconds})
    return races


if __name__ == "__main__":
    races = _prompt_races()
    if not races:
        print("No races entered, nothing to predict.")
        raise SystemExit(0)

    gender = input("\nGender (M/F, blank to skip): ").strip().upper() or None
    target_date_str = input("Target race date (YYYY-MM-DD, blank for today): ").strip() or None

    result = predict_5k(races, gender=gender, target_date=target_date_str)

    print(f"\nPredicted 5k time: {result['predicted_time']}  ({result['predicted_seconds']:.1f}s)")
    print(f"Riegel formula (from most recent race only): {result['riegel_time']}  ({result['riegel_seconds']:.1f}s)")
