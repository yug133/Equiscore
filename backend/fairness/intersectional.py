"""
Intersectional Fairness Module
Analyze fairness across intersectional subgroups: gender x region x occupation.
"""

from typing import Dict, List

import numpy as np
import pandas as pd


def create_intersectional_groups(
    df: pd.DataFrame,
    group_columns: List[str] = None,
) -> pd.Series:
    """
    Create intersectional subgroup labels by combining multiple sensitive
    attributes (e.g., gender × region × occupation_type).

    Produces 8 distinct subgroups for the EquiScore fairness analysis.

    Args:
        df: DataFrame containing sensitive attribute columns.
        group_columns: List of column names to intersect.
                       Default: ['CODE_GENDER', 'REGION_RATING_CLIENT', 'OCCUPATION_TYPE'].

    Returns:
        Series of intersectional group labels, one per applicant.
    """
    raise NotImplementedError("To be implemented")


def compute_intersectional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    intersectional_groups: pd.Series,
) -> Dict[str, Dict[str, float]]:
    """
    Compute fairness metrics (DPG, EOD, DIR) per intersectional subgroup.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        intersectional_groups: Series of intersectional group labels.

    Returns:
        Nested dictionary: {metric_name: {subgroup: value}}.
    """
    raise NotImplementedError("To be implemented")
