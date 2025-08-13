# model.py
# src/models/model.py

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from typing import Dict, Any

def build_model(model_name="RandomForest", params=None):
    """
    Build model berdasarkan nama dan parameter.
    
    Args:
        model_name (str): Nama model, contoh: "RandomForest" atau "XGBoost"
        params (dict): Hyperparameters untuk model.

    Returns:
        model: Model instance yang siap di-train.
    """
    if params is None:
        params = {}

    model_name = model_name.lower()

    if model_name == "randomforest":
        return RandomForestRegressor(**params)
    elif model_name == "xgboost":
        return XGBRegressor(**params)
    else:
        raise ValueError(f"Model '{model_name}' tidak dikenali. Gunakan 'RandomForest' atau 'XGBoost'.")