from fastapi import FastAPI, Request
import mlflow.sklearn
import pandas as pd
from prometheus_client import Counter, Histogram, start_http_server

# metrics
REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint", "method", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])

# load model from mlflow registry or local joblib
model = mlflow.sklearn.load_model("models:/HousePriceModel/Production")

app = FastAPI()

# start prometheus metrics server on port 8001 (or expose /metrics endpoint)
start_http_server(8001)

@app.post("/predict")
async def predict(payload: dict, request: Request):
    endpoint = "/predict"
    with REQUEST_LATENCY.labels(endpoint).time():
        try:
            df = pd.DataFrame([payload])
            preds = model.predict(df)
            REQUEST_COUNT.labels(endpoint, request.method, "200").inc()
            return {"prediction": float(preds[0])}
        except Exception as e:
            REQUEST_COUNT.labels(endpoint, request.method, "500").inc()
            raise