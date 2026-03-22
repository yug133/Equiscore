"""
Random Forest Ensemble Model
Ensemble baseline for credit default prediction.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 10,
    class_weight: str = "balanced",
    random_state: int = 42,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier as the ensemble baseline.

    Uses balanced class weights and limited depth to prevent overfitting
    on the imbalanced credit dataset.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector (0 = non-default, 1 = default).
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of each tree.
        class_weight: Strategy for class weighting ('balanced' or None).
        random_state: Random seed for reproducibility.

    Returns:
        Fitted RandomForestClassifier model instance.
    """
    raise NotImplementedError("To be implemented")


def predict_random_forest(
    model: RandomForestClassifier,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions and probability scores from the random forest model.

    Args:
        model: Fitted RandomForestClassifier instance.
        X: Feature matrix for prediction.

    Returns:
        Tuple of (predicted labels array, probability of default array).
    """
    raise NotImplementedError("To be implemented")
