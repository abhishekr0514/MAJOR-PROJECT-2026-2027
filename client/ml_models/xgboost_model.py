"""Tabular XGBoost Classifier Model for Heart Disease Risk Prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None  # type: ignore[assignment]
    XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    GradientBoostingClassifier = None  # type: ignore[assignment]
    SKLEARN_AVAILABLE = False


class TabularXGBoostModel:
    """XGBoost tabular classifier for clinical metrics (Age, BP, Cholesterol, Max HR)."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model: Any = None
        if XGBOOST_AVAILABLE and xgb is not None:
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
                eval_metric="logloss",
            )
        elif SKLEARN_AVAILABLE and GradientBoostingClassifier is not None:
            self.model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=random_state,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> TabularXGBoostModel:
        """Train tabular classifier model on clinical metrics matrix X and binary labels y."""
        if self.model is not None:
            self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary diagnostic target classes (0 for Low Risk, 1 for High Risk)."""
        if self.model is not None:
            return self.model.predict(X)
        # Fallback heuristic prediction if neither xgboost nor sklearn is installed
        return (X[:, 0] > 55).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict diagnostic risk probability scores of shape (N, 2)."""
        if self.model is not None and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        # Fallback probabilities
        p1 = np.clip(X[:, 0] / 100.0, 0.05, 0.95)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def save(self, filepath: str | Path) -> None:
        """Save model checkpoint to file path."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if XGBOOST_AVAILABLE and isinstance(self.model, xgb.XGBClassifier):
            self.model.save_model(str(path))
        elif hasattr(self.model, "save_model"):
            self.model.save_model(str(path))

    def load(self, filepath: str | Path) -> None:
        """Load trained model parameters from file path."""
        path = Path(filepath)
        if path.exists() and XGBOOST_AVAILABLE and isinstance(self.model, xgb.XGBClassifier):
            self.model.load_model(str(path))
