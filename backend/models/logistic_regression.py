import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_SAVE_PATH = Path(__file__).parent / "saved" / "logistic_regression.pkl"
SCALER_SAVE_PATH = Path(__file__).parent / "saved" / "logistic_scaler.pkl"


def build_logistic_model(
    max_iter: int = 1000,
    C: float = 1.0,
    class_weight: str = "balanced",
    random_state: int = 42
) -> LogisticRegression:
    """
    Build and return a configured Logistic Regression model.

    Uses balanced class weights to handle the heavy class imbalance
    in Home Credit dataset (~8% default rate). L2 regularization
    is applied by default via the C parameter.

    Args:
        max_iter: Maximum iterations for solver convergence.
        C: Inverse regularization strength. Smaller = stronger.
        class_weight: How to handle class imbalance. Default 'balanced'.
        random_state: Random seed for reproducibility.

    Returns:
        Configured but untrained LogisticRegression instance.
    """
    model = LogisticRegression(
        max_iter=max_iter,
        C=C,
        class_weight=class_weight,
        random_state=random_state,
        solver="lbfgs",
        n_jobs=-1
    )
    logger.info(f"Logistic Regression model built. C={C}, max_iter={max_iter}")
    return model


def train_logistic_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_features: bool = True
) -> Tuple[LogisticRegression, Optional[StandardScaler]]:
    """
    Train Logistic Regression on the training set.

    Optionally scales features using StandardScaler before fitting.
    Scaling is important for Logistic Regression as it is sensitive
    to feature magnitude differences (e.g. AMT_INCOME vs TRS score).

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series (binary 0/1).
        scale_features: Whether to apply StandardScaler. Default True.

    Returns:
        Tuple of (trained model, fitted scaler or None).
    """
    scaler = None
    X = X_train.copy()

    if scale_features:
        scaler = StandardScaler()
        X = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        logger.info("Features scaled with StandardScaler")

    model = build_logistic_model()
    logger.info(f"Training Logistic Regression on {X.shape[0]} samples...")
    model.fit(X, y_train)
    logger.info("Logistic Regression training complete")
    return model, scaler


def predict_proba_logistic(
    model: LogisticRegression,
    X_test: pd.DataFrame,
    scaler: Optional[StandardScaler] = None
) -> np.ndarray:
    """
    Generate default probability predictions for the test set.

    Returns the probability of class 1 (default) for each applicant.
    These raw probabilities are later scaled to the 300-900 score range
    by utils/score_scaler.py.

    Args:
        model: Trained LogisticRegression instance.
        X_test: Test feature DataFrame.
        scaler: Fitted StandardScaler if used during training.

    Returns:
        1D numpy array of default probabilities between 0 and 1.
    """
    X = X_test.copy()
    if scaler is not None:
        X = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
    proba = model.predict_proba(X)[:, 1]
    logger.info(f"Predictions generated for {len(proba)} applicants. "
                f"Mean prob={proba.mean():.4f}")
    return proba


def save_logistic_model(
    model: LogisticRegression,
    scaler: Optional[StandardScaler] = None
) -> None:
    """
    Save trained Logistic Regression model and scaler to disk.

    Creates the saved/ directory if it does not exist.
    Both model and scaler are serialized using pickle.

    Args:
        model: Trained LogisticRegression instance to save.
        scaler: Fitted StandardScaler to save (optional).
    """
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {MODEL_SAVE_PATH}")

    if scaler is not None:
        with open(SCALER_SAVE_PATH, "wb") as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to {SCALER_SAVE_PATH}")


def load_logistic_model() -> Tuple[LogisticRegression, Optional[StandardScaler]]:
    """
    Load trained Logistic Regression model and scaler from disk.

    Returns:
        Tuple of (loaded model, loaded scaler or None).

    Raises:
        FileNotFoundError: If model file does not exist at expected path.
    """
    if not MODEL_SAVE_PATH.exists():
        raise FileNotFoundError(f"No saved model at {MODEL_SAVE_PATH}")

    with open(MODEL_SAVE_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from {MODEL_SAVE_PATH}")

    scaler = None
    if SCALER_SAVE_PATH.exists():
        with open(SCALER_SAVE_PATH, "rb") as f:
            scaler = pickle.load(f)
        logger.info(f"Scaler loaded from {SCALER_SAVE_PATH}")

    return model, scaler