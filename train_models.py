import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

from build_dataset import riegel_predict

dataset = pd.read_csv("prediction_dataset.csv")

dataset["gender"] = dataset["gender"].fillna("U")
dataset["gender_M"] = (dataset["gender"] == "M").astype(int)
dataset["altitude"] = dataset["altitude"].fillna(dataset["altitude"].median())

BASE_FEATURES = ["prior_time_seconds", "prior_distance_m", "days_since_prior", "gender_M"]
NEW_FEATURES = ["race_count_so_far", "prior_is_5k", "best_prior_5k_equiv", "altitude"]

y = dataset["target_time_seconds"]

# split by athlete so the same athlete never appears in both train and test
splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(splitter.split(dataset, y, groups=dataset["athlete_id"]))

y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
test_set = dataset.iloc[test_idx]


def evaluate(name, pred):
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    return name, mae, rmse


def make_models():
    return {
        "Linear Regression": LinearRegression(),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
    }


riegel_pred = riegel_predict(test_set["prior_time_seconds"], test_set["prior_distance_m"])
riegel_mae = evaluate("Riegel formula", riegel_pred)[1]

# --- Feature ablation: how much does each new feature help XGBoost? ---
feature_sets = {
    "base (prior time/dist/days/gender)": BASE_FEATURES,
    "+ race_count_so_far, prior_is_5k": BASE_FEATURES + ["race_count_so_far", "prior_is_5k"],
    "+ best_prior_5k_equiv": BASE_FEATURES + ["best_prior_5k_equiv"],
    "+ altitude": BASE_FEATURES + ["altitude"],
    "all features": BASE_FEATURES + NEW_FEATURES,
}

print("=== Feature ablation (XGBoost) ===")
print(f"{'Feature set':<40}{'MAE (s)':>10}{'RMSE (s)':>10}")
print(f"{'Riegel formula (no features)':<40}{riegel_mae:>10.2f}")
ablation_models = {}
for label, cols in feature_sets.items():
    X_train, X_test = dataset[cols].iloc[train_idx], dataset[cols].iloc[test_idx]
    model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X_train, y_train)
    _, mae, rmse = evaluate(label, model.predict(X_test))
    print(f"{label:<40}{mae:>10.2f}{rmse:>10.2f}")
    ablation_models[label] = model
print()

# --- Full model comparison using the best feature set ---
best_features = feature_sets["all features"]
X_train, X_test = dataset[best_features].iloc[train_idx], dataset[best_features].iloc[test_idx]

results = [("Riegel formula", riegel_mae, evaluate("Riegel formula", riegel_pred)[2])]

trained = {}
for name, model in make_models().items():
    model.fit(X_train, y_train)
    trained[name] = model
    results.append(evaluate(name, model.predict(X_test)))

print("=== Full model comparison (all features) ===")
print(f"Train races: {len(X_train)}  Test races: {len(X_test)}")
print(f"Train athletes: {dataset.iloc[train_idx]['athlete_id'].nunique()}  "
      f"Test athletes: {dataset.iloc[test_idx]['athlete_id'].nunique()}")
print()
print(f"{'Model':<20}{'MAE (s)':>10}{'RMSE (s)':>10}")
for name, mae, rmse in results:
    print(f"{name:<20}{mae:>10.2f}{rmse:>10.2f}")
print()

for name, mae, _ in results[1:]:
    diff = (riegel_mae - mae) / riegel_mae * 100
    verb = "improves on" if diff > 0 else "underperforms"
    print(f"{name} {verb} Riegel by {abs(diff):.1f}% (MAE)")

print()
print("Linear Regression coefficients:")
for name, coef in zip(best_features, trained["Linear Regression"].coef_):
    print(f"  {name:<20} {coef:.4f}")
print(f"  {'intercept':<20} {trained['Linear Regression'].intercept_:.4f}")

for model_name in ["XGBoost", "Random Forest", "Extra Trees"]:
    print()
    print(f"{model_name} feature importances:")
    for name, imp in zip(best_features, trained[model_name].feature_importances_):
        print(f"  {name:<20} {imp:.4f}")
