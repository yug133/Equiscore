"""
Transaction Regularity Score (TRS) Feature
Measures how consistently an applicant transacts month to month.
"""

import pandas as pd
import numpy as np


def compute_transaction_regularity(
    installments_df: pd.DataFrame,
    window_months: int = 12,
) -> pd.Series:
    """
    Compute Transaction Regularity Score for each applicant.

    Measures how consistently an applicant transacts month to month.
    Formula: 1 - (std_dev of monthly payment counts / mean of monthly
    payment counts) over the last window_months.

    Higher score = more regular = stronger creditworthiness signal.

    Args:
        installments_df: Instalment payments table from Home Credit dataset.
        window_months: Number of months to compute regularity over.

    Returns:
        Series indexed by SK_ID_CURR with regularity score 0 to 1.
    """
    raise NotImplementedError("To be implemented")
