"""
Payment Behaviour Score (PBS) Feature
Measures timeliness and consistency of past payments.
"""

import pandas as pd
import numpy as np


def compute_payment_behaviour(
    installments_df: pd.DataFrame,
    pos_cash_df: pd.DataFrame,
) -> pd.Series:
    """
    Compute Payment Behaviour Score for each applicant.

    Evaluates repayment discipline by analyzing the ratio of on-time
    payments to total payments, combined with the average days of
    payment delay.

    Higher score = better payment behaviour = lower default risk.

    Args:
        installments_df: Instalment payments table with actual vs expected
                         payment dates.
        pos_cash_df: POS cash balance table with DPD (days past due) info.

    Returns:
        Series indexed by SK_ID_CURR with behaviour score 0 to 1.
    """
    raise NotImplementedError("To be implemented")
