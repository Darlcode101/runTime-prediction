# 5K Run Time Predictor

Predicts a runner's next 5K time from their race history using XGBoost, and benchmarks it against the Riegel formula (the standard rule of thumb runners use).

**Result:** 23% lower error than Riegel — mean absolute error of **49 s vs 64 s** on held-out athletes.

## Results

Trained on ~7,200 5K races from ~3,800 athletes. Train/test split is by athlete, so the model is scored on runners it has never seen.

| Model             | MAE (s) | vs. Riegel |
|-------------------|--------:|-----------:|
| Riegel formula    |   64.4  |          — |
| Linear Regression |   57.5  |     +10.8% |
| Random Forest     |   54.1  |     +16.1% |
| Extra Trees       |   53.8  |     +16.5% |
| **XGBoost**       | **49.4**|  **+23.4%** |

A feature ablation showed the biggest gains came from the athlete's best prior race (converted to a 5K equivalent), altitude, and race-day weather.

## How it works

- **Data cleaning** (`prepare_data.py`) — keeps athletes with 3+ races, at least one a 5K.
- **Features** (`features.py`) — built only from races *before* the target race, so there's no leakage. The same code is used for training and serving.
- **Experiments** (`train_models.py`) — model comparison and feature ablation.
- **Production model** (`train_final_model.py`) — XGBoost on the winning feature set.
- **App** (`app.py`, `predict.py`) — Flask web app and CLI.
- **Deployment** (`Dockerfile`, `deploy.sh`) — container image on AWS Lambda (arm64) via the Lambda Web Adapter, so the same Flask app runs locally and in the cloud.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000.

**Stack:** Python, pandas, scikit-learn, XGBoost, Flask, Docker, AWS Lambda/ECR
