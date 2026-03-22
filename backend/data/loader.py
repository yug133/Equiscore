"""
Data Loader Module
Load and join Home Credit CSV files for the EquiScore pipeline.
"""

import os
from typing import Dict, Optional

import pandas as pd


def load_application_data(data_dir: str) -> pd.DataFrame:
    """
    Load the main application_train.csv and application_test.csv files.

    Reads the primary Home Credit application tables containing
    socio-demographic and financial features for each loan application.

    Args:
        data_dir: Path to directory containing the raw CSV files.

    Returns:
        DataFrame with all application records.
    """
    raise NotImplementedError("To be implemented")


def load_bureau_data(data_dir: str) -> pd.DataFrame:
    """
    Load bureau.csv containing credit history from other financial institutions.

    Args:
        data_dir: Path to directory containing the raw CSV files.

    Returns:
        DataFrame with bureau credit records indexed by SK_ID_CURR.
    """
    raise NotImplementedError("To be implemented")


def load_installments_data(data_dir: str) -> pd.DataFrame:
    """
    Load installments_payments.csv with repayment history.

    Args:
        data_dir: Path to directory containing the raw CSV files.

    Returns:
        DataFrame with instalment payment records.
    """
    raise NotImplementedError("To be implemented")


def load_credit_card_data(data_dir: str) -> pd.DataFrame:
    """
    Load credit_card_balance.csv with monthly credit card snapshots.

    Args:
        data_dir: Path to directory containing the raw CSV files.

    Returns:
        DataFrame with credit card balance records.
    """
    raise NotImplementedError("To be implemented")


def load_pos_cash_data(data_dir: str) -> pd.DataFrame:
    """
    Load POS_CASH_balance.csv with monthly point-of-sale / cash loan snapshots.

    Args:
        data_dir: Path to directory containing the raw CSV files.

    Returns:
        DataFrame with POS cash balance records.
    """
    raise NotImplementedError("To be implemented")


def load_previous_applications(data_dir: str) -> pd.DataFrame:
    """
    Load previous_application.csv with historical loan applications.

    Args:
        data_dir: Path to directory containing the raw CSV files.

    Returns:
        DataFrame with previous application records.
    """
    raise NotImplementedError("To be implemented")


def load_all_tables(data_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load all Home Credit CSV tables and return as a dictionary.

    Convenience function that loads all six tables and returns them
    in a single dictionary keyed by table name.

    Args:
        data_dir: Path to directory containing the raw CSV files.

    Returns:
        Dictionary mapping table names to their DataFrames.
    """
    raise NotImplementedError("To be implemented")
