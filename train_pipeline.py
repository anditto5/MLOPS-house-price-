
# train_pipeline.py
import mlflow
import mlflow.sklearn
import logging
from pythonjsonlogger import jsonlogger
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_diabetes

# ==============================
# Setup structured logging
# ==============================
# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_pipeline():
    logger.info("Starting training pipeline...")

    # Enable autologging
    mlflow.sklearn.autolog()

# ==============================
# Load Data
# ==============================
logger.info("Loading dataset...")
df = pd.read_csv("C:/Users/gito2/Downloads/house_price_prediction/Artifacts/train.csv")

target = "SalePrice"
y = df[target]
X = df.drop(columns=[target])

# ==============================
# Preprocessing
# ==============================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ==============================
# Model + Pipeline
# ==============================
model = RandomForestRegressor(random_state=42)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10]
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# MLflow Tracking
# ==============================
mlflow.set_experiment("house-prices-pipeline")

with mlflow.start_run():
    logger.info("Starting model training with GridSearchCV...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error"
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)

    # Metrics (handle old sklearn)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Logging to console
    logger.info({
        "event": "training_metrics",
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "best_params": grid_search.best_params_
    })

    # Logging to MLflow
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Save model to MLflow
    mlflow.sklearn.log_model(best_model, "model")
    logger.info("Model saved to MLflow.")

logger.info("Training pipeline completed.")

