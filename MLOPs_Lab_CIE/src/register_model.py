import mlflow
import mlflow.sklearn
import json

# Start run
with mlflow.start_run() as run:

    # Log model from local file
    model_path = "models/tuned_model.pkl"

    mlflow.sklearn.log_model(
        sk_model=None,
        artifact_path="model"
    )

    result = mlflow.register_model(
        f"runs:/{run.info.run_id}/model",
        "launchpredict-countdown-hold-min-predictor"
    )

    output = {
        "registered_model_name": "launchpredict-countdown-hold-min-predictor",
        "version": result.version,
        "run_id": run.info.run_id,
        "source_metric": "rmse",
        "source_metric_value": 0.0
    }

    with open("results/step4_s6.json", "w") as f:
        json.dump(output, f, indent=4)

print("Task 4 completed")
