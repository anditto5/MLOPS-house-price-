# tracking.py
import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifactp

def setup_mlflow(experiment_name="house_price_prediction", tracking_uri="mlruns"):
    """
    Setup MLflow experiment tracking.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

def start_run(run_name):
    """
    Start MLflow run.
    """
    return mlflow.start_run(run_name=run_name)

def log_params(params: dict):
    """
    Log parameters.
    """
    for key, value in params.items():
        mlflow.log_param(key, value)

def log_metrics(metrics: dict):
    """
    Log metrics.
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

def log_model(model, artifact_path="model"):
    """
    Log trained model.
    """
    mlflow.sklearn.log_model(model, artifact_path)