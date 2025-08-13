# trainer.py
# train_with_mlflow.py
import time
import pandas as pd
import structlog
import mlflow
import mlflow.sklearn
from logging_config import setup_logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

setup_logging()
log = structlog.get_logger()

# Set tracking URI (lokal)
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("house-prices")

# ===== Tahap Preprocessing =====
log.info("preprocessing_start")
df = pd.read_csv("train.csv")
log.info(
    "data_loaded",
    shape=df.shape,
    missing_values=df.isnull().sum().sum()
)

df = df.fillna(0)
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]
log.info("preprocessing_done", shape=X.shape)

# ===== Training Model + MLflow Logging =====
with mlflow.start_run() as run:
    start_time = time.time()
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X, y)
    train_time = round(time.time() - start_time, 2)

    log.info(
        "model_training_done",
        model_type="RandomForestRegressor",
        n_estimators=200,
        max_depth=10,
        train_time_seconds=train_time
    )

    # Logging parameter
    mlflow.log_params({
        "n_estimators": 200,
        "max_depth": 10,
        "random_state": 42
    })

    # Evaluasi
    y_pred = model.predict(X)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    log.info(
        "model_performance",
        RMSE=rmse,
        MAE=mae,
        R2=r2
    )

    # Logging metrik
    mlflow.log_metrics({
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    })

    # Simpan model sebagai artifact
    mlflow.sklearn.log_model(model, artifact_path="model")

log.info("training_completed", run_id=run.info.run_id)