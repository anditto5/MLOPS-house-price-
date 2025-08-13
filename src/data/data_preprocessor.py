# data_preprocessor.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from utils.logger import get_logger

logger = get_logger(__name__)

def preprocess_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Preprocessing data:
    - Scaling untuk kolom numerik
    - One-hot encoding untuk kolom kategorikal
    - Logging terstruktur
    """
    
    # Deteksi kolom numerik & kategorikal
    numerical_cols = [col for col in X_train.columns if X_train[col].dtype in [np.float64, np.int64]]
    categorical_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
    

    logger.info("Identified feature types", extra={
        "extra_data": {
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols
        }
    })

    # Pipeline untuk numerik
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Pipeline untuk kategorikal
    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Gabungkan ke ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    logger.info("Fitting preprocessing pipeline", extra={
        "extra_data": {
            "num_features": len(numerical_cols),
            "cat_features": len(categorical_cols)
        }
    })

    # Fit di train dan transform di train & test
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    logger.info("Finished preprocessing", extra={
        "extra_data": {
            "train_shape": X_train_processed.shape,
            "test_shape": X_test_processed.shape
        }
    })

    return X_train_processed, X_test_processed, preprocessor
