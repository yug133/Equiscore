"""
Standard XGBoost Model
Performance benchmark for credit default prediction.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict = None,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 20,
    X_val: pd.DataFrame = None,
    y_val: pd.Series = None,
) -> xgb.Booster:
    """
    Train a standard XGBoost model as the performance benchmark.

    Uses GPU-accelerated training if available, with early stopping
    on a validation set to prevent overfitting.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector (0 = non-default, 1 = default).
        params: XGBoost hyperparameters dict. If None, uses defaults
                optimized for binary classification with imbalance.
        num_boost_round: Maximum number of boosting rounds.
        early_stopping_rounds: Rounds without improvement before stopping.
        X_val: Optional validation feature matrix for early stopping.
        y_val: Optional validation target vector.

    Returns:
        Trained xgb.Booster instance.
    """
    raise NotImplementedError("To be implemented")


def predict_xgboost(
    model: xgb.Booster,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions and probability scores from the XGBoost model.

    Args:
        model: Trained xgb.Booster instance.
        X: Feature matrix for prediction.

    Returns:
        Tuple of (predicted labels array, probability of default array).
    """
    raise NotImplementedError("To be implemented")
