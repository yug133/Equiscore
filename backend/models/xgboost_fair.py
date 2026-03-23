import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Optional
from xgboost import XGBClassifier
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from utils.logger import get_logger

logger = get_logger(__name__)

MODEL_SAVE_PATH = Path(__file__).parent / "saved" / "xgboost_fair.pkl"

CONSTRAINT_DEMOGRAPHIC_PARITY = "demographic_parity"
CONSTRAINT_EQUALIZED_ODDS = "equalized_odds"


def build_base_xgboost_for_fairness(
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    scale_pos_weight: float = 11.8,
    random_state: int = 42
) -> XGBClassifier:
    """
    Build a base XGBoost estimator to be wrapped by ExponentiatedGradient.

    Uses slightly shallower trees (max_depth=5) compared to standard
    XGBoost to reduce complexity, since ExponentiatedGradient trains
    multiple copies of this base estimator during fairness optimization.

    Args:
        n_estimators: Number of boosting rounds per estimator.
        max_depth: Maximum tree depth. Default 5.
        learning_rate: Step size shrinkage. Default 0.05.
        scale_pos_weight: Weight ratio for positive class.
        random_state: Random seed for reproducibility.

    Returns:
        Configured but untrained XGBClassifier instance.
    """
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="auc",
        use_label_encoder=False,
        n_jobs=-1,
        verbosity=0
    )
    logger.info(f"Base XGBoost for fairness built. max_depth={max_depth}")
    return model


def build_fair_xgboost(
    constraint: str = CONSTRAINT_EQUALIZED_ODDS,
    eps: float = 0.02
) -> ExponentiatedGradient:
    """
    Build a fairness-constrained XGBoost using Fairlearn ExponentiatedGradient.

    ExponentiatedGradient is a reduction approach that wraps a standard
    ML estimator and trains it to satisfy a fairness constraint while
    minimizing prediction error.

    Two constraint options:
    - DemographicParity: Equal approval rates across sensitive groups.
    - EqualizedOdds: Equal TPR and FPR across sensitive groups.
      Preferred for credit scoring.

    Args:
        constraint: Fairness constraint type. Either
                    'demographic_parity' or 'equalized_odds'.
        eps: Maximum allowed constraint violation. Default 0.02.

    Returns:
        Configured but untrained ExponentiatedGradient instance.
    """
    base_model = build_base_xgboost_for_fairness()

    if constraint == CONSTRAINT_DEMOGRAPHIC_PARITY:
        fairness_constraint = DemographicParity(difference_bound=eps)
    else:
        fairness_constraint = EqualizedOdds(difference_bound=eps)

    fair_model = ExponentiatedGradient(
        estimator=base_model,
        constraints=fairness_constraint,
        eps=eps
    )
    logger.info(f"Fair XGBoost built. Constraint={constraint}, eps={eps}")
    return fair_model


def train_fair_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sensitive_features: pd.Series,
    constraint: str = CONSTRAINT_EQUALIZED_ODDS,
    eps: float = 0.02
) -> ExponentiatedGradient:
    """
    Train fairness-constrained XGBoost on the training set.

    The sensitive_features Series defines which demographic group
    each applicant belongs to. ExponentiatedGradient will optimize
    the model to satisfy the fairness constraint across these groups.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target Series (binary 0/1).
        sensitive_features: Series of group labels per applicant.
        constraint: Fairness constraint type. Default 'equalized_odds'.
        eps: Maximum allowed fairness violation. Default 0.02.

    Returns:
        Trained ExponentiatedGradient instance.
    """
    fair_model = build_fair_xgboost(constraint, eps)
    logger.info(f"Training Fair XGBoost on {X_train.shape[0]} samples...")
    logger.info(f"Sensitive feature groups: "
                f"{sensitive_features.value_counts().to_dict()}")
    fair_model.fit(X_train, y_train, sensitive_features=sensitive_features)
    logger.info("Fair XGBoost training complete")
    return fair_model


def predict_proba_fair_xgboost(
    model: ExponentiatedGradient,
    X_test: pd.DataFrame
) -> np.ndarray:
    """
    Generate fairness-constrained default probability predictions.

    ExponentiatedGradient uses a randomized mixture of base estimators.
    It exposes _pmf_predict() for soft probabilities. We use that to
    get class 1 probabilities, falling back to hard predict() if needed.

    Args:
        model: Trained ExponentiatedGradient instance.
        X_test: Test feature DataFrame.

    Returns:
        1D numpy array of default probabilities between 0 and 1.
    """
    try:
        # ExponentiatedGradient exposes _pmf_predict for probabilities
        proba_matrix = model._pmf_predict(X_test)
        proba = proba_matrix[:, 1]
        logger.info(f"Used _pmf_predict for Fair XGBoost probabilities")
    except Exception as e:
        logger.warning(f"_pmf_predict failed ({e}), using hard predict fallback")
        proba = model.predict(X_test).astype(float)

    logger.info(f"Fair XGBoost predictions generated for {len(proba)} "
                f"applicants. Mean prob={proba.mean():.4f}")
    return proba


def save_fair_xgboost(model: ExponentiatedGradient) -> None:
    """
    Save trained Fair XGBoost model to disk using pickle.

    Args:
        model: Trained ExponentiatedGradient instance to save.
    """
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Fair XGBoost saved to {MODEL_SAVE_PATH}")


def load_fair_xgboost() -> ExponentiatedGradient:
    """
    Load trained Fair XGBoost model from disk.

    Returns:
        Loaded ExponentiatedGradient instance.

    Raises:
        FileNotFoundError: If no saved model exists at expected path.
    """
    if not MODEL_SAVE_PATH.exists():
        raise FileNotFoundError(f"No saved model at {MODEL_SAVE_PATH}")
    with open(MODEL_SAVE_PATH, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Fair XGBoost loaded from {MODEL_SAVE_PATH}")
    return model