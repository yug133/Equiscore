"""
Logistic Regression Baseline Model
Simple interpretable baseline for credit default prediction.
"""

from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a logistic regression model as the interpretable baseline.

    Uses balanced class weights to handle the imbalanced target distribution.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector (0 = non-default, 1 = default).
        class_weight: Strategy for class weighting ('balanced' or None).
        random_state: Random seed for reproducibility.

    Returns:
        Fitted LogisticRegression model instance.
    """
    raise NotImplementedError("To be implemented")


def predict_logistic_regression(
    model: LogisticRegression,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions and probability scores from the logistic regression model.

    Args:
        model: Fitted LogisticRegression instance.
        X: Feature matrix for prediction.

    Returns:
        Tuple of (predicted labels array, probability of default array).
    """
    raise NotImplementedError("To be implemented")
