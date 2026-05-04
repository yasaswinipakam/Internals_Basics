import pandas as pd
import numpy as np
import json
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
data = pd.read_csv("data/training_data.csv")

# Split features and target
X = data.drop("countdown_hold_min", axis=1)
y = data["countdown_hold_min"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow experiment
mlflow.set_experiment("launchpredict-countdown-hold-min")

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge()
}

results = []
best_rmse = float("inf")
best_model = None
best_name = ""

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model", name)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.set_tag("experiment_type", "baseline_comparison")

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model
            best_name = name

# Save best model
joblib.dump(best_model, "models/best_model.pkl")

# Save JSON result
output = {
    "experiment_name": "launchpredict-countdown-hold-min",
    "models": results,
    "best_model": best_name,
    "best_metric_name": "rmse",
    "best_metric_value": best_rmse
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 1 completed")
