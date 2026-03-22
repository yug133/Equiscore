"""
Data Preprocessor Module
Handle missing values, encoding, and feature transformations.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Impute missing values in numeric and categorical columns.

    Numeric columns are filled using the specified strategy (mean/median).
    Categorical columns are filled with the mode.

    Args:
        df: Input DataFrame with potential missing values.
        strategy: Imputation strategy for numeric columns ('mean' or 'median').

    Returns:
        DataFrame with missing values imputed.
    """
    raise NotImplementedError("To be implemented")


def encode_categorical_features(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical features using label encoding.

    Automatically detects object-type columns if none are specified.
    Stores the encoders for inverse transformation during inference.

    Args:
        df: Input DataFrame with categorical columns.
        columns: Optional list of column names to encode. If None, all
                 object-type columns are encoded.

    Returns:
        Tuple of (encoded DataFrame, dict of column -> LabelEncoder).
    """
    raise NotImplementedError("To be implemented")


def scale_numeric_features(
    df: pd.DataFrame, columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize numeric features using StandardScaler.

    Args:
        df: Input DataFrame with numeric columns.
        columns: Optional list of numeric column names to scale.
                 If None, all numeric columns are scaled.

    Returns:
        Tuple of (scaled DataFrame, fitted StandardScaler instance).
    """
    raise NotImplementedError("To be implemented")


def drop_high_null_columns(
    df: pd.DataFrame, threshold: float = 0.4
) -> pd.DataFrame:
    """
    Drop columns where the fraction of missing values exceeds the threshold.

    Args:
        df: Input DataFrame.
        threshold: Maximum fraction of null values allowed (0.0 to 1.0).

    Returns:
        DataFrame with high-null columns removed.
    """
    raise NotImplementedError("To be implemented")


def preprocess_pipeline(
    df: pd.DataFrame,
    null_threshold: float = 0.4,
    impute_strategy: str = "median",
) -> Tuple[pd.DataFrame, dict]:
    """
    Run the full preprocessing pipeline: drop high-null columns,
    impute remaining missing values, and encode categorical features.

    Args:
        df: Raw input DataFrame.
        null_threshold: Column drop threshold for missing values.
        impute_strategy: Strategy for numeric imputation.

    Returns:
        Tuple of (preprocessed DataFrame, metadata dict with encoders/scalers).
    """
    raise NotImplementedError("To be implemented")
