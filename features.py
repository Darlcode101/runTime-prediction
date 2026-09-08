"""
Shared feature engineering, used by both build_dataset.py (training) and
predict.py (serving), so a prediction is built from the exact same logic
that produced the training data.
"""

TARGET_DISTANCE_M = 5000


def riegel_predict(t1, d1, d2=TARGET_DISTANCE_M, exponent=1.06):
    return t1 * (d2 / d1) ** exponent


def compute_features(prior_races, target_date, gender=None, altitude=None,
                      temperature=None, wind_speed=None, humidity=None):
    """
    prior_races: list of dicts with keys 'date', 'distance_m', 'time_seconds' -
                 every race the athlete ran before target_date. Order doesn't matter,
                 this sorts them itself.
    target_date: date of the 5k being predicted.
    Returns a dict of feature_name -> value matching prediction_dataset.csv's columns.
    """
    prior = sorted(prior_races, key=lambda r: r["date"])
    last = prior[-1]
    has_second_prior = len(prior) >= 2
    second_last = prior[-2] if has_second_prior else None

    prior_5k_equivs = [riegel_predict(r["time_seconds"], r["distance_m"]) for r in prior]

    if has_second_prior:
        last_equiv = riegel_predict(last["time_seconds"], last["distance_m"])
        second_equiv = riegel_predict(second_last["time_seconds"], second_last["distance_m"])
        trend_5k_equiv = last_equiv - second_equiv
        days_between_last_two = (last["date"] - second_last["date"]).days
    else:
        trend_5k_equiv = None
        days_between_last_two = None

    return {
        "prior_time_seconds": last["time_seconds"],
        "prior_distance_m": last["distance_m"],
        "days_since_prior": (target_date - last["date"]).days,
        "gender_M": 1 if gender == "M" else 0,
        "race_count_so_far": len(prior),
        "prior_is_5k": int(last["distance_m"] == TARGET_DISTANCE_M),
        "best_prior_5k_equiv": min(prior_5k_equivs),
        "altitude": altitude,
        "has_second_prior": int(has_second_prior),
        "trend_5k_equiv": trend_5k_equiv,
        "days_between_last_two": days_between_last_two,
        "temperature": temperature,
        "wind_speed": wind_speed,
        "humidity": humidity,
    }
