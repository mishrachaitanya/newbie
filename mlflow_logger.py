# mlflow_logger.py
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import mlflow.sklearn
import joblib
import json
import os
import warnings
import argparse

# ============================
# Command-line arguments
# ============================
parser = argparse.ArgumentParser(description="Log clean and poisoned models to MLflow.")
parser.add_argument("--mlflow_ip", type=str, required=True, help="MLflow server IP (with port), e.g., 35.194.31.99:8200")
args = parser.parse_args()
TRACKING_URI = f"http://{args.mlflow_ip}"

# ============================
# Paths
# ============================
EXPERIMENT_NAME = "Churn-Prediction-Clean-vs-Poisoned"
CLEAN_MODEL_PATH = "model_clean.joblib"
POISONED_MODEL_PATH = "model_poisoned.joblib"
CLEAN_METRICS_PATH = "metrics_clean.json"
POISONED_METRICS_PATH = "metrics_poisoned.json"

# ============================
# Setup MLflow
# ============================
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient(TRACKING_URI)

# Get or create experiment
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if exp is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = exp.experiment_id

print(f"[INFO] Using MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"[INFO] Experiment ID: {experiment_id}")

# ============================
# Helper to log one run
# ============================
def log_run(run_name, model_path, metrics_path, tag):
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        print(f"[INFO] Logging run: {run_name} (ID: {run.info.run_id})")

        # Log metrics
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(metrics_path)

        # Log model
        if os.path.exists(model_path):
            model = joblib.load(model_path)

            # Attempt to infer signature (skip if fails)
            signature = None
            try:
                if hasattr(model, "n_features_in_"):
                    X_example = [[0] * model.n_features_in_]
                    y_example = model.predict(X_example)
                    signature = infer_signature(X_example, y_example)
            except Exception as e:
                warnings.warn(f"Could not infer signature for {run_name}: {e}")

            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name=f"{run_name}-model"
            )

        # Set tag
        mlflow.set_tag("dataset", tag)

        print(f"[INFO] Finished logging {run_name}")
        print(f"🏃 View run at: {mlflow.get_tracking_uri()}/#/experiments/{experiment_id}/runs/{run.info.run_id}")

# ============================
# Log Clean Model
# ============================
if os.path.exists(CLEAN_MODEL_PATH) and os.path.exists(CLEAN_METRICS_PATH):
    log_run("clean", CLEAN_MODEL_PATH, CLEAN_METRICS_PATH, "clean")
else:
    print(f"[WARN] Missing files for clean model: {CLEAN_MODEL_PATH}, {CLEAN_METRICS_PATH}")

# ============================
# Log Poisoned Model
# ============================
if os.path.exists(POISONED_MODEL_PATH) and os.path.exists(POISONED_METRICS_PATH):
    log_run("poisoned", POISONED_MODEL_PATH, POISONED_METRICS_PATH, "poisoned")
else:
    print(f"[WARN] Missing files for poisoned model: {POISONED_MODEL_PATH}, {POISONED_METRICS_PATH}")

print("[INFO] ✅ Logging complete.")

