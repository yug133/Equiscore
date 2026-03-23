import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


def compute_monthly_transaction_counts(
    installments_df: pd.DataFrame,
    window_months: int = 12
) -> pd.DataFrame:
    """
    Compute the number of transactions per applicant per month.

    Groups installments by SK_ID_CURR and payment month, counting
    how many payments were made each month within the window period.

    Args:
        installments_df: Instalment payments table from Home Credit.
                         Must contain SK_ID_CURR, DAYS_INSTALMENT.
        window_months: Number of most recent months to consider.

    Returns:
        DataFrame with columns: SK_ID_CURR, month, transaction_count.
    """
    df = installments_df.copy()

    # Convert DAYS_INSTALMENT to month buckets (negative days from today)
    df["month"] = (df["DAYS_INSTALMENT"] // 30).astype(int)

    # Keep only the most recent window_months
    max_month = df["month"].max()
    cutoff = max_month - window_months
    df = df[df["month"] >= cutoff]

    monthly_counts = (
        df.groupby(["SK_ID_CURR", "month"])
        .size()
        .reset_index(name="transaction_count")
    )

    logger.info(f"Monthly transaction counts computed for {monthly_counts['SK_ID_CURR'].nunique()} applicants")
    return monthly_counts


def compute_transaction_regularity(
    installments_df: pd.DataFrame,
    window_months: int = 12
) -> pd.Series:
    """
    Compute Transaction Regularity Score (TRS) for each applicant.

    Measures how consistently an applicant transacts month to month.
    Formula: 1 - (std_dev of monthly transaction counts / mean of
    monthly transaction counts) over the last window_months.

    A score close to 1 means very regular transactions.
    A score close to 0 means highly irregular transaction behaviour.
    Applicants with only 1 month of data get a score of 0.5 (neutral).

    Args:
        installments_df: Instalment payments table from Home Credit.
                         Must contain SK_ID_CURR, DAYS_INSTALMENT.
        window_months: Number of months to compute regularity over.

    Returns:
        Series indexed by SK_ID_CURR with TRS score between 0 and 1.
    """
    monthly_counts = compute_monthly_transaction_counts(
        installments_df, window_months
    )

    def regularity_score(counts: pd.Series) -> float:
        if len(counts) <= 1:
            return 0.5  # Neutral score for insufficient history
        mean_val = counts.mean()
        std_val = counts.std()
        if mean_val == 0:
            return 0.0
        cv = std_val / mean_val  # Coefficient of variation
        return float(np.clip(1 - cv, 0, 1))

    trs = (
        monthly_counts
        .groupby("SK_ID_CURR")["transaction_count"]
        .apply(regularity_score)
        .rename("TRS")
    )

    logger.info(f"TRS computed for {len(trs)} applicants. "
                f"Mean={trs.mean():.4f}, Std={trs.std():.4f}")
    return trs


def flag_irregular_applicants(
    trs: pd.Series,
    threshold: float = 0.3
) -> pd.Series:
    """
    Flag applicants with very low transaction regularity.

    Applicants below the threshold are considered to have
    irregular financial behaviour, which is a risk signal.

    Args:
        trs: Series of TRS scores indexed by SK_ID_CURR.
        threshold: Score below which applicant is flagged. Default 0.3.

    Returns:
        Boolean Series indexed by SK_ID_CURR.
        True = irregular (flagged), False = regular.
    """
    flagged = trs < threshold
    logger.info(f"Flagged {flagged.sum()} applicants as irregular "
                f"(threshold={threshold})")
    return flagged