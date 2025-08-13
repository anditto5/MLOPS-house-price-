from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List
import mlflow
import mlflow.sklearn
import pandas as pd
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML API with FastAPI, MLflow, and Logging")

# ==============================
# Request/Response Schemas
# ==============================
class TrainRequest(BaseModel):
    n_estimators: int = Field(100, description="Number of trees in RandomForest")
    max_depth: int = Field(5, description="Max depth of trees")
    min_samples_split: int = Field(2, description="Minimum number of samples required to split an internal node")
    min_samples_leaf: int = Field(1, description="Minimum number of samples required to be at a leaf node")


class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="List of numerical features for prediction")


class PredictionResponse(BaseModel):
    prediction: float


# ==============================
# Training Endpoint
# ==============================
@app.post("/train")
def train_model(request: TrainRequest):
    try:
        logger.info(f"Starting training with params: {request.dict()}")

        # Example training data
        df = pd.DataFrame({
            "X1": [1, 2, 3, 4, 5],
            "X2": [5, 4, 3, 2, 1],
            "y": [2, 4, 6, 8, 10]
        })
        X = df[["X1", "X2"]]
        y = df["y"]

        with mlflow.start_run():
            # Log all parameters
            for param_name, param_value in request.dict().items():
                mlflow.log_param(param_name, param_value)

            model = RandomForestRegressor(
                n_estimators=request.n_estimators,
                max_depth=request.max_depth,
                min_samples_split=request.min_samples_split,
                min_samples_leaf=request.min_samples_leaf,
                random_state=42
            )
            model.fit(X, y)

            preds = model.predict(X)
            mse = mean_squared_error(y, preds)
            rmse = mean_squared_error(y, preds, squared=False)

            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)

            # Save to MLflow registry
            mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="house_price_model")

        logger.info(f"Training completed with MSE: {mse}, RMSE: {rmse}")
        return {"status": "Model trained", "mse": mse, "rmse": rmse}

    except Exception as e:
        logger.exception("Error during training")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# Prediction Endpoint
# ==============================
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        logger.info(f"Received prediction request: {request.features}")

        model_uri = "models:/house_price_model/Production"
        model = mlflow.sklearn.load_model(model_uri)

        input_df = pd.DataFrame([request.features])
        pred = model.predict(input_df)[0]

        logger.info(f"Prediction result: {pred}")
        return PredictionResponse(prediction=float(pred))

    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# Error Handler
# ==============================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException: {exc.detail}")
    return {"error": exc.detail}
