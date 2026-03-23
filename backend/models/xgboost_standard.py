import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict
from xgboost import XGBClassifier
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_SAVE_PATH = Path(__file__).parent / "saved" / "xgboost_standard.pkl"

# Scale positive weight handles class imbalance
# Formula: count(negative) / count(positive)
# For Home Credit: ~282000 / ~24000 ≈ 11.8
DEFAULT_SCALE_POS_WEIGHT = 11.8


def build_xgboost_model(
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    scale_pos_weight: float = DEFAULT_SCALE_POS_WEIGHT,
    random_state: int = 42
) -> XGBClassifier:
    """
    Build and return a configured standard XGBoost classifier.

    scale_pos_weight handles severe class imbalance by upweighting
    the minority class (defaulters) during training.
    subsample and colsample_bytree add stochasticity to reduce
    overfitting on the training set.

    Args:
        n_estimators: Number of boosting rounds. Default 500.
        max_depth: Maximum tree depth. Default 6.
        learning_rate: Step size shrinkage. Default 0.05.
        subsample: Fraction of samples per tree. Default 0.8.
        colsample_bytree: Fraction of features per tree. Default 0.8.
        scale_pos_weight: Weight ratio for positive class. Default 11.8.
        random_state: Random seed for reproducibility.

    Returns:
        Configured but untrained XGBClassifier instance.
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="auc",
        use_label_encoder=False,
        n_jobs=-1,
        verbosity=0
    )
    logger.info(f"XGBoost model built. n_estimators={n_estimators}, "
                f"max_depth={max_depth}, lr={learning_rate}")
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    early_stopping_rounds: int = 50
) -> XGBClassifier:
    """
    Train XGBoost with optional early stopping on validation set.

    If validation data is provided, early stopping is used to
    halt training when AUC stops improving, preventing overfitting.
    Without validation data, trains for the full n_estimators rounds.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series (binary 0/1).
        X_val: Optional validation feature DataFrame for early stopping.
        y_val: Optional validation target Series for early stopping.
        early_stopping_rounds: Stop if no improvement after N rounds.

    Returns:
        Trained XGBClassifier instance.
    """
    model = build_xgboost_model()

    fit_params = {}
    if X_val is not None and y_val is not None:
        fit_params["eval_set"] = [(X_val, y_val)]
        fit_params["early_stopping_rounds"] = early_stopping_rounds
        fit_params["verbose"] = 50
        logger.info("Training XGBoost with early stopping...")
    else:
        logger.info("Training XGBoost without early stopping...")

    logger.info(f"Training on {X_train.shape[0]} samples, "
                f"{X_train.shape[1]} features...")
    model.fit(X_train, y_train, **fit_params)
    logger.info("XGBoost training complete")
    return model


def predict_proba_xgboost(
    model: XGBClassifier,
    X_test: pd.DataFrame
) -> np.ndarray:
    """
    Generate default probability predictions using XGBoost.

    Returns probability of class 1 (default) for each applicant.
    XGBoost probabilities are well-calibrated and can be directly
    used as input to the score scaler.

    Args:
        model: Trained XGBClassifier instance.
        X_test: Test feature DataFrame.

    Returns:
        1D numpy array of default probabilities between 0 and 1.
    """
    proba = model.predict_proba(X_test)[:, 1]
    logger.info(f"XGBoost predictions generated for {len(proba)} applicants. "
                f"Mean prob={proba.mean():.4f}")
    return proba


def get_feature_importances(
    model: XGBClassifier,
    feature_names: list
) -> pd.DataFrame:
    """
    Extract and rank feature importances from trained XGBoost model.

    Uses gain-based importance which measures the average gain
    of splits using each feature — more informative than frequency.

    Args:
        model: Trained XGBClassifier instance.
        feature_names: List of feature column names matching X_train.

    Returns:
        DataFrame with columns: feature, importance.
        Sorted by importance descending.
    """
    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info(f"Top 5 XGBoost features:\n{importances.head()}")
    return importances


def save_xgboost(model: XGBClassifier) -> None:
    """
    Save trained XGBoost model to disk using pickle.

    Args:
        model: Trained XGBClassifier instance to save.
    """
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"XGBoost model saved to {MODEL_SAVE_PATH}")


def load_xgboost() -> XGBClassifier:
    """
    Load trained XGBoost model from disk.

    Returns:
        Loaded XGBClassifier instance.

    Raises:
        FileNotFoundError: If no saved model exists at expected path.
    """
    if not MODEL_SAVE_PATH.exists():
        raise FileNotFoundError(f"No saved model at {MODEL_SAVE_PATH}")
    with open(MODEL_SAVE_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"XGBoost model loaded from {MODEL_SAVE_PATH}")
    return model