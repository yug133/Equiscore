import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from utils.logger import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "TARGET"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features (X) and target (y) from the main DataFrame.

    The target column TARGET in Home Credit dataset is binary:
    0 = loan repaid on time, 1 = loan defaulted.

    Args:
        df: Preprocessed application DataFrame containing target column.
        target_col: Name of the target column. Default is 'TARGET'.

    Returns:
        Tuple of (X: feature DataFrame, y: target Series).

    Raises:
        KeyError: If target_col is not found in the DataFrame.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    logger.info(f"Target distribution:\n{y.value_counts(normalize=True)}")
    return X, y


def stratified_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified 80-20 train-test split.

    Stratification ensures both train and test sets maintain the same
    class ratio as the full dataset. This is critical for imbalanced
    datasets like Home Credit where only ~8% of applicants default.

    Args:
        X: Feature DataFrame.
        y: Target Series (binary: 0 or 1).
        test_size: Fraction of data for test set. Default is 0.2 (20%).
        random_state: Random seed for reproducibility. Default is 42.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    logger.info(f"Splitting data: {1-test_size:.0%} train / {test_size:.0%} test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    logger.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    logger.info(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")
    logger.info(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")
    return X_train, X_test, y_train, y_test


def get_split_summary(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Return a summary DataFrame describing the train-test split.

    Shows row counts, default rates, and feature counts for both
    train and test sets. Useful for verifying stratification worked.

    Args:
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.

    Returns:
        Summary DataFrame with split statistics.
    """
    summary = pd.DataFrame([
        {
            "split": "train",
            "rows": len(X_train),
            "features": X_train.shape[1],
            "default_rate": round(y_train.mean(), 4),
            "default_count": int(y_train.sum()),
        },
        {
            "split": "test",
            "rows": len(X_test),
            "features": X_test.shape[1],
            "default_rate": round(y_test.mean(), 4),
            "default_count": int(y_test.sum()),
        }
    ])
    logger.info(f"Split summary:\n{summary}")
    return summary


def run_full_split_pipeline(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Run the complete split pipeline in one call.

    Convenience wrapper that calls split_features_target followed
    by stratified_train_test_split. Use this in the main pipeline.

    Args:
        df: Fully preprocessed application DataFrame.
        target_col: Name of the target column. Default is 'TARGET'.
        test_size: Fraction for test set. Default is 0.2.
        random_state: Random seed. Default is 42.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X, y = split_features_target(df, target_col)
    return stratified_train_test_split(X, y, test_size, random_state)