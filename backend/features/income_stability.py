"""
Income Stability Index (ISI) Feature
Measures consistency of income-to-credit ratio over time.
"""

import pandas as pd
import numpy as np


def compute_income_stability(
    application_df: pd.DataFrame,
    bureau_df: pd.DataFrame,
) -> pd.Series:
    """
    Compute Income Stability Index for each applicant.

    Calculates the coefficient of variation of income-to-credit ratio
    across the applicant's credit history. Lower variation indicates
    more stable income relative to credit usage.

    Formula: 1 - CV(income_credit_ratio) clamped to [0, 1].

    Args:
        application_df: Main application table with AMT_INCOME_TOTAL.
        bureau_df: Bureau table with historical credit amounts.

    Returns:
        Series indexed by SK_ID_CURR with stability score 0 to 1.
    """
    raise NotImplementedError("To be implemented")
