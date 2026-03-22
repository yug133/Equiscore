"""
Data Splitter Module
Train-test split with stratified sampling for imbalanced credit data.
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pd.DataFrame,
    target_column: str = "TARGET",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets using stratified sampling.

    Preserves the class distribution of the target variable (default/non-default)
    in both the train and test sets to handle class imbalance.

    Args:
        df: Preprocessed DataFrame with features and target column.
        target_column: Name of the binary target column.
        test_size: Fraction of data to reserve for testing (0.0 to 1.0).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df) with stratified target distribution.
    """
    raise NotImplementedError("To be implemented")


def get_feature_target_split(
    df: pd.DataFrame,
    target_column: str = "TARGET",
    exclude_columns: list = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) from target (y), optionally excluding ID columns.

    Args:
        df: Input DataFrame containing both features and target.
        target_column: Name of the target column.
        exclude_columns: List of column names to exclude from features
                         (e.g., ['SK_ID_CURR']).

    Returns:
        Tuple of (X features DataFrame, y target Series).
    """
    raise NotImplementedError("To be implemented")
