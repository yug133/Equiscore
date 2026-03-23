import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shap
from xgboost import XGBClassifier
from utils.logger import get_logger

logger = get_logger(__name__)

SHAP_SAVE_PATH = Path(__file__).parent.parent / "models" / "saved" / "shap_explainer.pkl"


def build_shap_explainer(
    model: XGBClassifier,
    X_background: pd.DataFrame
) -> shap.TreeExplainer:
    """
    Build a SHAP TreeExplainer for the trained XGBoost model.

    TreeExplainer is the most efficient SHAP method for tree-based
    models. It computes exact Shapley values rather than approximations,
    making it suitable for credit scoring where explanation accuracy
    is critical for regulatory compliance.

    Args:
        model: Trained XGBClassifier or ExponentiatedGradient instance.
               For Fair XGBoost, pass the best estimator.
        X_background: Background dataset for SHAP baseline computation.
                      Use a sample of 100-500 rows for efficiency.

    Returns:
        Configured shap.TreeExplainer instance.
    """
    logger.info(f"Building SHAP TreeExplainer on "
                f"{X_background.shape[0]} background samples...")
    explainer = shap.TreeExplainer(
        model,
        data=X_background,
        feature_perturbation="interventional"
    )
    logger.info("SHAP TreeExplainer built successfully")
    return explainer


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Compute SHAP values for a set of applicants.

    SHAP values represent the contribution of each feature to the
    model's prediction for each applicant. Positive values push
    the prediction towards default (class 1), negative values
    push towards repayment (class 0).

    Args:
        explainer: Fitted shap.TreeExplainer instance.
        X: Feature DataFrame for which to compute SHAP values.
           Can be a single row (one applicant) or multiple rows.

    Returns:
        2D numpy array of shape (n_samples, n_features) with SHAP
        values for class 1 (default probability).
    """
    logger.info(f"Computing SHAP values for {X.shape[0]} applicants...")
    shap_values = explainer.shap_values(X)

    # For binary classification, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    logger.info(f"SHAP values computed. Shape: {shap_values.shape}")
    return shap_values


def get_global_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Compute global feature importance from SHAP values.

    Global importance = mean absolute SHAP value per feature across
    all applicants. This tells the loan officer which features drive
    credit decisions most strongly across the entire portfolio.

    Args:
        shap_values: 2D array of SHAP values (n_samples, n_features).
        feature_names: List of feature column names.
        top_n: Number of top features to return. Default 20.

    Returns:
        DataFrame with columns: feature, mean_abs_shap.
        Sorted by importance descending.
    """
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False).head(top_n)

    logger.info(f"Top 5 global features:\n{importance.head()}")
    return importance.reset_index(drop=True)


def get_individual_shap_explanation(
    explainer: shap.TreeExplainer,
    X_single: pd.DataFrame,
    top_n: int = 10
) -> Dict:
    """
    Compute individual SHAP explanation for a single applicant.

    Returns a waterfall-ready dictionary showing which features
    pushed the applicant's score up or down from the baseline.
    This is the data that powers the SHAP waterfall chart in the
    loan officer dashboard.

    Args:
        explainer: Fitted shap.TreeExplainer instance.
        X_single: Single-row DataFrame for one applicant.
        top_n: Number of top features to include. Default 10.

    Returns:
        Dictionary with keys:
        - base_value: Model baseline prediction (float)
        - feature_contributions: List of dicts with feature, value,
          shap_value, direction (positive=risk, negative=safe)
        - top_risk_features: Top N features increasing default risk
        - top_safe_features: Top N features decreasing default risk
    """
    shap_vals = compute_shap_values(explainer, X_single)[0]
    feature_names = X_single.columns.tolist()
    feature_values = X_single.iloc[0].tolist()

    contributions = pd.DataFrame({
        "feature": feature_names,
        "feature_value": feature_values,
        "shap_value": shap_vals,
        "direction": ["risk" if v > 0 else "safe" for v in shap_vals]
    }).sort_values("shap_value", key=abs, ascending=False)

    result = {
        "base_value": float(explainer.expected_value
                            if not isinstance(explainer.expected_value, np.ndarray)
                            else explainer.expected_value[1]),
        "feature_contributions": contributions.head(top_n).to_dict("records"),
        "top_risk_features": contributions[
            contributions["shap_value"] > 0
        ].head(top_n // 2).to_dict("records"),
        "top_safe_features": contributions[
            contributions["shap_value"] < 0
        ].head(top_n // 2).to_dict("records"),
    }

    logger.info(f"Individual SHAP explanation computed. "
                f"Top risk: {contributions.iloc[0]['feature']}, "
                f"SHAP={contributions.iloc[0]['shap_value']:.4f}")
    return result


def save_shap_explainer(explainer: shap.TreeExplainer) -> None:
    """
    Save fitted SHAP explainer to disk using pickle.

    Args:
        explainer: Fitted shap.TreeExplainer instance to save.
    """
    SHAP_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SHAP_SAVE_PATH, "wb") as f:
        pickle.dump(explainer, f)
    logger.info(f"SHAP explainer saved to {SHAP_SAVE_PATH}")


def load_shap_explainer() -> shap.TreeExplainer:
    """
    Load fitted SHAP explainer from disk.

    Returns:
        Loaded shap.TreeExplainer instance.

    Raises:
        FileNotFoundError: If no saved explainer exists.
    """
    if not SHAP_SAVE_PATH.exists():
        raise FileNotFoundError(f"No saved explainer at {SHAP_SAVE_PATH}")
    with open(SHAP_SAVE_PATH, "rb") as f:
        explainer = pickle.load(f)
    logger.info(f"SHAP explainer loaded from {SHAP_SAVE_PATH}")
    return explainer