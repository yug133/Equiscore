"""
SHAP Explainer Module
Global and individual-level SHAP explanations using TreeExplainer.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import shap


def create_shap_explainer(model: Any) -> shap.TreeExplainer:
    """
    Create a SHAP TreeExplainer for the given tree-based model.

    Args:
        model: Trained tree-based model (XGBoost, Random Forest, etc.).

    Returns:
        Initialized shap.TreeExplainer instance.
    """
    raise NotImplementedError("To be implemented")


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Compute SHAP values for all samples in the feature matrix.

    Args:
        explainer: Fitted SHAP TreeExplainer.
        X: Feature matrix to explain.

    Returns:
        Array of SHAP values with shape (n_samples, n_features).
    """
    raise NotImplementedError("To be implemented")


def get_global_feature_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Compute global feature importance as mean absolute SHAP values.

    Args:
        shap_values: SHAP values array (n_samples, n_features).
        feature_names: List of feature column names.

    Returns:
        Dictionary mapping feature name to mean |SHAP| value, sorted descending.
    """
    raise NotImplementedError("To be implemented")


def get_individual_explanation(
    shap_values: np.ndarray,
    feature_names: List[str],
    sample_index: int,
    base_value: float,
) -> Dict[str, Any]:
    """
    Generate an individual waterfall explanation for a single applicant.

    Returns the SHAP contributions for each feature along with the base
    value and final prediction, suitable for rendering a waterfall chart.

    Args:
        shap_values: SHAP values array (n_samples, n_features).
        feature_names: List of feature column names.
        sample_index: Index of the sample to explain.
        base_value: Expected value (base prediction) from the explainer.

    Returns:
        Dictionary with 'base_value', 'features' (list of feature contributions),
        and 'output_value' keys.
    """
    raise NotImplementedError("To be implemented")
