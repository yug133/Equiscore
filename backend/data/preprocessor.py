import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

# Columns to drop due to high missingness or data leakage
HIGH_MISSING_THRESHOLD = 0.5

CATEGORICAL_COLUMNS = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START", "ORGANIZATION_TYPE", "REGION_RATING_CLIENT"
]

NUMERICAL_COLUMNS = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "CNT_FAM_MEMBERS", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"
]


def drop_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = HIGH_MISSING_THRESHOLD
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns where the fraction of missing values exceeds threshold.

    Columns with more than 50% missing values are unlikely to contribute
    meaningful signal and may harm model generalization.

    Args:
        df: Input DataFrame to clean.
        threshold: Maximum allowed fraction of missing values (default 0.5).

    Returns:
        Tuple of (cleaned DataFrame, list of dropped column names).
    """
    missing_frac = df.isnull().mean()
    cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()
    logger.info(f"Dropping {len(cols_to_drop)} high-missing columns: {cols_to_drop}")
    return df.drop(columns=cols_to_drop), cols_to_drop


def impute_numerical(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    strategy: str = "median"
) -> pd.DataFrame:
    """
    Impute missing values in numerical columns using median or mean.

    Median imputation is preferred for skewed financial data as it is
    robust to outliers (e.g. AMT_INCOME_TOTAL, AMT_CREDIT).

    Args:
        df: Input DataFrame.
        columns: List of numerical columns to impute. Defaults to
                 NUMERICAL_COLUMNS if None.
        strategy: Either 'median' or 'mean'. Default is 'median'.

    Returns:
        DataFrame with numerical missing values filled.
    """
    cols = columns or NUMERICAL_COLUMNS
    cols_present = [c for c in cols if c in df.columns]
    for col in cols_present:
        if strategy == "median":
            fill_value = df[col].median()
        else:
            fill_value = df[col].mean()
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            logger.info(f"Imputing {col}: {missing_count} missing with {strategy}={fill_value:.4f}")
        df[col] = df[col].fillna(fill_value)
    return df


def impute_categorical(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Impute missing values in categorical columns using mode (most frequent).

    For categorical features like OCCUPATION_TYPE or NAME_INCOME_TYPE,
    filling with mode preserves the original distribution.

    Args:
        df: Input DataFrame.
        columns: List of categorical columns to impute. Defaults to
                 CATEGORICAL_COLUMNS if None.

    Returns:
        DataFrame with categorical missing values filled.
    """
    cols = columns or CATEGORICAL_COLUMNS
    cols_present = [c for c in cols if c in df.columns]
    for col in cols_present:
        mode_val = df[col].mode()[0]
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            logger.info(f"Imputing {col}: {missing_count} missing with mode='{mode_val}'")
        df[col] = df[col].fillna(mode_val)
    return df


def encode_categoricals(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    One-hot encode categorical columns using pandas get_dummies.

    Binary columns like CODE_GENDER and FLAG_OWN_CAR are label-encoded
    (0/1). Multi-class columns like OCCUPATION_TYPE are one-hot encoded
    with drop_first=True to avoid multicollinearity.

    Args:
        df: Input DataFrame.
        columns: List of categorical columns to encode. Defaults to
                 CATEGORICAL_COLUMNS if None.

    Returns:
        DataFrame with categorical columns replaced by encoded versions.
    """
    cols = columns or CATEGORICAL_COLUMNS
    cols_present = [c for c in cols if c in df.columns]
    logger.info(f"One-hot encoding {len(cols_present)} categorical columns...")
    df = pd.get_dummies(df, columns=cols_present, drop_first=True)
    return df


def fix_anomalous_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix known anomalous values in the Home Credit dataset.

    DAYS_EMPLOYED has a known anomaly: value 365243 is used as a
    placeholder for unemployed applicants. Replace with NaN then
    re-impute. Similarly clip extreme outliers in income columns.

    Args:
        df: Input DataFrame (should be application_train or test).

    Returns:
        DataFrame with anomalous values corrected.
    """
    if "DAYS_EMPLOYED" in df.columns:
        anomaly_count = (df["DAYS_EMPLOYED"] == 365243).sum()
        logger.info(f"Fixing DAYS_EMPLOYED anomaly: {anomaly_count} rows set to NaN")
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    if "AMT_INCOME_TOTAL" in df.columns:
        upper = df["AMT_INCOME_TOTAL"].quantile(0.99)
        clipped = (df["AMT_INCOME_TOTAL"] > upper).sum()
        logger.info(f"Clipping AMT_INCOME_TOTAL at 99th percentile ({upper}): {clipped} rows")
        df["AMT_INCOME_TOTAL"] = df["AMT_INCOME_TOTAL"].clip(upper=upper)

    return df


def preprocess_application(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline for the main application table.

    Runs all preprocessing steps in the correct order:
    1. Fix anomalous values
    2. Drop high-missing columns
    3. Impute numerical columns
    4. Impute categorical columns
    5. Encode categoricals

    Args:
        df: Raw application_train or application_test DataFrame.

    Returns:
        Fully preprocessed DataFrame ready for feature engineering.
    """
    logger.info(f"Starting preprocessing. Shape: {df.shape}")
    df = fix_anomalous_values(df)
    df, _ = drop_high_missing_columns(df)
    df = impute_numerical(df)
    df = impute_categorical(df)
    df = encode_categoricals(df)
    logger.info(f"Preprocessing complete. Shape: {df.shape}")
    return df