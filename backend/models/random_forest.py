import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict
from sklearn.ensemble import RandomForestClassifier
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_SAVE_PATH = Path(__file__).parent / "saved" / "random_forest.pkl"


def build_random_forest_model(
    n_estimators: int = 300,
    max_depth: int = 10,
    min_samples_leaf: int = 50,
    class_weight: str = "balanced",
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Build and return a configured Random Forest classifier.

    Uses balanced class weights to handle Home Credit class imbalance.
    min_samples_leaf=50 prevents overfitting on the minority class.
    max_depth=10 controls tree complexity for better generalization.

    Args:
        n_estimators: Number of trees in the forest. Default 300.
        max_depth: Maximum depth of each tree. Default 10.
        min_samples_leaf: Minimum samples required at a leaf node.
        class_weight: Class balancing strategy. Default 'balanced'.
        random_state: Random seed for reproducibility.

    Returns:
        Configured but untrained RandomForestClassifier instance.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    logger.info(f"Random Forest built. n_estimators={n_estimators}, "
                f"max_depth={max_depth}")
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestClassifier:
    """
    Train Random Forest on the training set.

    No feature scaling required for tree-based models.
    Training may take several minutes depending on dataset size
    and n_estimators setting.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series (binary 0/1).

    Returns:
        Trained RandomForestClassifier instance.
    """
    model = build_random_forest_model()
    logger.info(f"Training Random Forest on {X_train.shape[0]} samples "
                f"with {X_train.shape[1]} features...")
    model.fit(X_train, y_train)
    logger.info("Random Forest training complete")
    return model


def predict_proba_random_forest(
    model: RandomForestClassifier,
    X_test: pd.DataFrame
) -> np.ndarray:
    """
    Generate default probability predictions using Random Forest.

    Returns probability of class 1 (default) for each applicant.
    These probabilities are later converted to 300-900 credit scores.

    Args:
        model: Trained RandomForestClassifier instance.
        X_test: Test feature DataFrame.

    Returns:
        1D numpy array of default probabilities between 0 and 1.
    """
    proba = model.predict_proba(X_test)[:, 1]
    logger.info(f"Random Forest predictions generated for {len(proba)} "
                f"applicants. Mean prob={proba.mean():.4f}")
    return proba


def get_feature_importances(
    model: RandomForestClassifier,
    feature_names: list
) -> pd.DataFrame:
    """
    Extract and rank feature importances from the trained Random Forest.

    Uses mean decrease in impurity (MDI) as the importance metric.
    Results are sorted descending so most important features appear first.

    Args:
        model: Trained RandomForestClassifier instance.
        feature_names: List of feature column names matching X_train.

    Returns:
        DataFrame with columns: feature, importance.
        Sorted by importance descending.
    """
    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info(f"Top 5 features:\n{importances.head()}")
    return importances


def save_random_forest(model: RandomForestClassifier) -> None:
    """
    Save trained Random Forest model to disk using pickle.

    Creates the saved/ directory if it does not exist.

    Args:
        model: Trained RandomForestClassifier instance to save.
    """
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Random Forest saved to {MODEL_SAVE_PATH}")


def load_random_forest() -> RandomForestClassifier:
    """
    Load trained Random Forest model from disk.

    Returns:
        Loaded RandomForestClassifier instance.

    Raises:
        FileNotFoundError: If no saved model exists at expected path.
    """
    if not MODEL_SAVE_PATH.exists():
        raise FileNotFoundError(f"No saved model at {MODEL_SAVE_PATH}")
    with open(MODEL_SAVE_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Random Forest loaded from {MODEL_SAVE_PATH}")
    return model