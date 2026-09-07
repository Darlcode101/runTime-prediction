import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

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

# --- Riegel baseline (no training needed, just formula) ---
riegel_pred = riegel_predict(test_set["prior_time_seconds"], test_set["prior_distance_m"])
riegel_mae = mean_absolute_error(y_test, riegel_pred)
riegel_rmse = np.sqrt(mean_squared_error(y_test, riegel_pred))

# --- Linear Regression ---
model = LinearRegression()
model.fit(X_train, y_train)
lr_pred = model.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

print(f"Train races: {len(X_train)}  Test races: {len(X_test)}")
print(f"Train athletes: {dataset.iloc[train_idx]['athlete_id'].nunique()}  "
      f"Test athletes: {dataset.iloc[test_idx]['athlete_id'].nunique()}")
print()
print(f"{'Model':<20}{'MAE (s)':>10}{'RMSE (s)':>10}")
print(f"{'Riegel formula':<20}{riegel_mae:>10.2f}{riegel_rmse:>10.2f}")
print(f"{'Linear Regression':<20}{lr_mae:>10.2f}{lr_rmse:>10.2f}")
print()
improvement = (riegel_mae - lr_mae) / riegel_mae * 100
print(f"Linear Regression {'improves on' if improvement > 0 else 'underperforms'} "
      f"Riegel by {abs(improvement):.1f}% (MAE)")
print()
print("Learned coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"  {name:<20} {coef:.4f}")
print(f"  {'intercept':<20} {model.intercept_:.4f}")
