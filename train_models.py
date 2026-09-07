import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBRegressor

from build_dataset import riegel_predict

dataset = pd.read_csv("prediction_dataset.csv")

dataset["gender"] = dataset["gender"].fillna("U")
dataset["gender_M"] = (dataset["gender"] == "M").astype(int)

feature_cols = ["prior_time_seconds", "prior_distance_m", "days_since_prior", "gender_M"]
X = dataset[feature_cols]
y = dataset["target_time_seconds"]

# split by athlete so the same athlete never appears in both train and test
splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups=dataset["athlete_id"]))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
test_set = dataset.iloc[test_idx]


def evaluate(name, pred):
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    return name, mae, rmse


results = []

# --- Riegel baseline (no training needed, just formula) ---
riegel_pred = riegel_predict(test_set["prior_time_seconds"], test_set["prior_distance_m"])
results.append(evaluate("Riegel formula", riegel_pred))

# --- Linear Regression ---
lr = LinearRegression()
lr.fit(X_train, y_train)
results.append(evaluate("Linear Regression", lr.predict(X_test)))

# --- XGBoost ---
xgb = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
xgb.fit(X_train, y_train)
results.append(evaluate("XGBoost", xgb.predict(X_test)))

print(f"Train races: {len(X_train)}  Test races: {len(X_test)}")
print(f"Train athletes: {dataset.iloc[train_idx]['athlete_id'].nunique()}  "
      f"Test athletes: {dataset.iloc[test_idx]['athlete_id'].nunique()}")
print()
print(f"{'Model':<20}{'MAE (s)':>10}{'RMSE (s)':>10}")
for name, mae, rmse in results:
    print(f"{name:<20}{mae:>10.2f}{rmse:>10.2f}")
print()

riegel_mae = results[0][1]
for name, mae, _ in results[1:]:
    diff = (riegel_mae - mae) / riegel_mae * 100
    verb = "improves on" if diff > 0 else "underperforms"
    print(f"{name} {verb} Riegel by {abs(diff):.1f}% (MAE)")

print()
print("Linear Regression coefficients:")
for name, coef in zip(feature_cols, lr.coef_):
    print(f"  {name:<20} {coef:.4f}")
print(f"  {'intercept':<20} {lr.intercept_:.4f}")

print()
print("XGBoost feature importances:")
for name, imp in zip(feature_cols, xgb.feature_importances_):
    print(f"  {name:<20} {imp:.4f}")
