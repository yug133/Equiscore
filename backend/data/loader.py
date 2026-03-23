import os
import pandas as pd
from typing import Dict
from utils.logger import get_logger

logger = get_logger(__name__)

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "raw")

REQUIRED_FILES = {
    "application_train": "application_train.csv",
    "application_test": "application_test.csv",
    "installments": "installments_payments.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "credit_card": "credit_card_balance.csv",
    "pos_cash": "POS_CASH_balance.csv",
}


def load_all_tables(data_path: str = RAW_DATA_PATH) -> Dict[str, pd.DataFrame]:
    """
    Load all Home Credit CSV files into a dictionary of DataFrames.

    Reads each required CSV file from the given directory and returns
    them as a keyed dictionary for downstream use in the pipeline.

    Args:
        data_path: Path to the folder containing raw CSV files.

    Returns:
        Dictionary mapping table names to their DataFrames.
        Keys: application_train, application_test, installments,
              bureau, bureau_balance, credit_card, pos_cash
    
    Raises:
        FileNotFoundError: If any required CSV file is missing.
    """
    dataframes = {}
    for key, filename in REQUIRED_FILES.items():
        filepath = os.path.join(data_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing required file: {filepath}")
        logger.info(f"Loading {filename}...")
        dataframes[key] = pd.read_csv(filepath)
        logger.info(f"Loaded {filename}: {dataframes[key].shape}")
    return dataframes


def load_single_table(filename: str, data_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load a single CSV file by filename.

    Useful for loading one table at a time during development
    or feature engineering without loading the full dataset.

    Args:
        filename: Name of the CSV file (e.g. 'application_train.csv')
        data_path: Path to the folder containing raw CSV files.

    Returns:
        DataFrame with the contents of the CSV file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    filepath = os.path.join(data_path, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    logger.info(f"Loading {filename}...")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {filename}: {df.shape}")
    return df


def get_data_summary(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Print a summary of all loaded DataFrames.

    Shows row count, column count, and memory usage for each table.
    Useful for a quick sanity check after loading.

    Args:
        dataframes: Dictionary of table name to DataFrame.

    Returns:
        Summary DataFrame with shape and memory info per table.
    """
    summary = []
    for name, df in dataframes.items():
        summary.append({
            "table": name,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2)
        })
    return pd.DataFrame(summary)