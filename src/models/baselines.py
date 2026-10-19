import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaselineModels:
    """
    Implements classical machine learning baselines (Logistic Regression, 
    Random Forest, XGBoost) using numerical rolling window features.
    """
    
    def __init__(self, model_type: str = "xgboost", **kwargs):
        self.model_type = model_type.lower()
        self.model = self._initialize_model(**kwargs)

    def _initialize_model(self, **kwargs):
        if self.model_type == "logistic_regression":
            return LogisticRegression(max_iter=1000, random_state=42, **kwargs)
        elif self.model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42, **kwargs)
        elif self.model_type == "xgboost":
            # XGBoost classifier for multi-class (0: DOWN, 1: NEUTRAL, 2: UP)
            return XGBClassifier(eval_metric='mlogloss', random_state=42, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. Choose from 'logistic_regression', 'random_forest', 'xgboost'.")

    def _flatten_windows(self, X: np.ndarray) -> np.ndarray:
        """
        Flattens rolling window tensor of shape (num_samples, window_size, 5) 
        into tabular feature matrix of shape (num_samples, window_size * 5).
        """
        if len(X.shape) == 3:
            return X.reshape(X.shape[0], -1)
        return X

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Trains the baseline model on chronological training data."""
        X_flat = self._flatten_windows(X_train)
        logging.info(f"Training baseline [{self.model_type}] on {X_flat.shape[0]} samples with {X_flat.shape[1]} features...")
        self.model.fit(X_flat, y_train)
        logging.info(f"Successfully trained baseline [{self.model_type}].")

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generates class predictions (0, 1, 2) for test data."""
        X_flat = self._flatten_windows(X_test)
        return self.model.predict(X_flat)

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """Generates class probability distributions for UP/DOWN/NEUTRAL."""
        X_flat = self._flatten_windows(X_test)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_flat)
        raise NotImplementedError(f"Model {self.model_type} does not support probability predictions.")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluates model performance against test data."""
        preds = self.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1_weighted = f1_score(y_test, preds, average='weighted', zero_division=0)
        
        logging.info(f"Baseline [{self.model_type}] Evaluation -> Accuracy: {acc:.4f} | Weighted F1: {f1_weighted:.4f}")
        return {
            "accuracy": float(acc),
            "f1_weighted": float(f1_weighted),
            "classification_report": classification_report(y_test, preds, zero_division=0)
        }