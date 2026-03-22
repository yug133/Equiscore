"""
Feature Pipeline Module
Combines all five engineered features into a single feature matrix.
"""

from typing import Dict

import pandas as pd

from features.transaction_regularity import compute_transaction_regularity
from features.income_stability import compute_income_stability
from features.payment_behaviour import compute_payment_behaviour
from features.digital_footprint import compute_digital_footprint
from features.geo_income_index import compute_geo_income_index


def build_feature_matrix(
    tables: Dict[str, pd.DataFrame],
    window_months: int = 12,
) -> pd.DataFrame:
    """
    Build the complete engineered feature matrix by computing all five
    custom features and merging them with the base application data.

    Features computed:
        - TRS: Transaction Regularity Score
        - ISI: Income Stability Index
        - PBS: Payment Behaviour Score
        - DFS: Digital Footprint Score
        - GII: Geo-Income Index

    Args:
        tables: Dictionary of DataFrames from data.loader.load_all_tables().
        window_months: Lookback window for time-based features.

    Returns:
        DataFrame indexed by SK_ID_CURR with all five engineered features
        appended to the original application features.
    """
    raise NotImplementedError("To be implemented")


def get_feature_names() -> list:
    """
    Return the list of engineered feature column names.

    Returns:
        List of feature name strings: ['TRS', 'ISI', 'PBS', 'DFS', 'GII'].
    """
    raise NotImplementedError("To be implemented")
