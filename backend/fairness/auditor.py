"""
Fairness Auditor Module
Compute fairness metrics using Fairlearn MetricFrame.
"""

from typing import Dict

import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame


def compute_demographic_parity_gap(
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
) -> Dict[str, float]:
    """
    Compute Demographic Parity Gap (DPG) per subgroup.

    DPG measures the difference in positive prediction rates (acceptance
    rates) between the most and least favoured subgroups.

    Args:
        y_pred: Predicted binary labels.
        sensitive_features: Series of sensitive attribute values.

    Returns:
        Dictionary with 'overall_dpg' and per-subgroup acceptance rates.
    """
    raise NotImplementedError("To be implemented")


def compute_equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
) -> Dict[str, float]:
    """
    Compute Equalized Odds Difference (EOD) per subgroup.

    EOD measures the maximum difference in true positive rates and
    false positive rates across subgroups.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        sensitive_features: Series of sensitive attribute values.

    Returns:
        Dictionary with 'eod_tpr', 'eod_fpr', and per-subgroup rates.
    """
    raise NotImplementedError("To be implemented")


def compute_disparate_impact_ratio(
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
) -> Dict[str, float]:
    """
    Compute Disparate Impact Ratio (DIR) per subgroup.

    DIR = min(subgroup acceptance rate) / max(subgroup acceptance rate).
    A ratio close to 1.0 indicates fairness; below 0.8 is often flagged.

    Args:
        y_pred: Predicted binary labels.
        sensitive_features: Series of sensitive attribute values.

    Returns:
        Dictionary with 'overall_dir' and per-subgroup acceptance rates.
    """
    raise NotImplementedError("To be implemented")


def run_full_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Run the complete fairness audit: DPG, EOD, and DIR for all sensitive features.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        sensitive_features: DataFrame of sensitive attribute columns.

    Returns:
        Nested dictionary: {metric_name: {subgroup: value}}.
    """
    raise NotImplementedError("To be implemented")
