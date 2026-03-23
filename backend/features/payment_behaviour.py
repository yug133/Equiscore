import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

# Thresholds for payment behaviour classification
LATE_PAYMENT_DAYS_THRESHOLD = 5      # Days late before considered a late payment
SEVERE_LATE_DAYS_THRESHOLD = 30      # Days late before considered severely late
EARLY_PAYMENT_BONUS = 0.1            # Bonus score for paying early


def compute_payment_delays(
    installments_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute the delay in days for each payment made by each applicant.

    Delay = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT
    Positive delay = paid late
    Negative delay = paid early
    Zero = paid exactly on time

    Args:
        installments_df: Instalment payments table from Home Credit.
                         Must contain SK_ID_CURR, DAYS_INSTALMENT,
                         DAYS_ENTRY_PAYMENT.

    Returns:
        DataFrame with original columns plus new 'payment_delay' column.
    """
    df = installments_df.copy()
    df["payment_delay"] = df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]
    late_count = (df["payment_delay"] > LATE_PAYMENT_DAYS_THRESHOLD).sum()
    early_count = (df["payment_delay"] < 0).sum()
    logger.info(f"Payment delays computed. Late: {late_count}, Early: {early_count}")
    return df


def compute_late_payment_rate(
    installments_df: pd.DataFrame,
    threshold_days: int = LATE_PAYMENT_DAYS_THRESHOLD
) -> pd.Series:
    """
    Compute the fraction of late payments per applicant.

    A payment is considered late if it was made more than
    threshold_days after the scheduled instalment date.

    Args:
        installments_df: Instalment payments table from Home Credit.
        threshold_days: Number of days grace period before flagging
                        as late. Default is 5 days.

    Returns:
        Series indexed by SK_ID_CURR with late payment rate 0 to 1.
        0 = never late, 1 = always late.
    """
    df = compute_payment_delays(installments_df)
    df["is_late"] = (df["payment_delay"] > threshold_days).astype(int)

    late_rate = (
        df.groupby("SK_ID_CURR")["is_late"]
        .mean()
        .rename("late_payment_rate")
    )

    logger.info(f"Late payment rate computed for {len(late_rate)} applicants. "
                f"Mean late rate: {late_rate.mean():.4f}")
    return late_rate


def compute_severe_late_count(
    installments_df: pd.DataFrame,
    severe_threshold: int = SEVERE_LATE_DAYS_THRESHOLD
) -> pd.Series:
    """
    Count the number of severely late payments per applicant.

    Severely late payments (30+ days) are a stronger default signal
    than minor delays and should be weighted more heavily in PBS.

    Args:
        installments_df: Instalment payments table from Home Credit.
        severe_threshold: Days late to qualify as severe. Default 30.

    Returns:
        Series indexed by SK_ID_CURR with count of severe late payments.
    """
    df = compute_payment_delays(installments_df)
    df["is_severe_late"] = (df["payment_delay"] > severe_threshold).astype(int)

    severe_counts = (
        df.groupby("SK_ID_CURR")["is_severe_late"]
        .sum()
        .rename("severe_late_count")
    )

    logger.info(f"Severe late counts computed. "
                f"Max: {severe_counts.max()}, Mean: {severe_counts.mean():.4f}")
    return severe_counts


def compute_early_payment_rate(
    installments_df: pd.DataFrame
) -> pd.Series:
    """
    Compute the fraction of payments made early per applicant.

    Early payments indicate proactive financial behaviour and
    are used as a positive signal in the PBS formula.

    Args:
        installments_df: Instalment payments table from Home Credit.
                         Must contain SK_ID_CURR, DAYS_INSTALMENT,
                         DAYS_ENTRY_PAYMENT.

    Returns:
        Series indexed by SK_ID_CURR with early payment rate 0 to 1.
    """
    df = compute_payment_delays(installments_df)
    df["is_early"] = (df["payment_delay"] < 0).astype(int)

    early_rate = (
        df.groupby("SK_ID_CURR")["is_early"]
        .mean()
        .rename("early_payment_rate")
    )

    logger.info(f"Early payment rate computed for {len(early_rate)} applicants. "
                f"Mean early rate: {early_rate.mean():.4f}")
    return early_rate


def compute_payment_behaviour_score(
    installments_df: pd.DataFrame,
    late_weight: float = 0.6,
    severe_weight: float = 0.3,
    early_bonus_weight: float = 0.1
) -> pd.Series:
    """
    Compute the Payment Behaviour Score (PBS) for each applicant.

    Combines late payment rate, severe late payment penalty, and
    early payment bonus into a single score between 0 and 1.

    Formula:
        PBS = 1
            - (late_weight * late_payment_rate)
            - (severe_weight * clipped_severe_rate)
            + (early_bonus_weight * early_payment_rate)
        clipped to [0, 1]

    Higher PBS = better payment behaviour = lower credit risk.

    Args:
        installments_df: Instalment payments table from Home Credit.
        late_weight: Weight for late payment rate penalty. Default 0.6.
        severe_weight: Weight for severe late payment penalty. Default 0.3.
        early_bonus_weight: Weight for early payment bonus. Default 0.1.

    Returns:
        Series indexed by SK_ID_CURR with PBS score between 0 and 1.
    """
    late_rate = compute_late_payment_rate(installments_df)
    severe_counts = compute_severe_late_count(installments_df)
    early_rate = compute_early_payment_rate(installments_df)

    # Normalize severe counts to 0-1 range
    severe_rate = (severe_counts / severe_counts.max()).fillna(0)

    # Align all series on SK_ID_CURR index
    combined = pd.DataFrame({
        "late_rate": late_rate,
        "severe_rate": severe_rate,
        "early_rate": early_rate
    }).fillna(0)

    combined["PBS"] = (
        1
        - (late_weight * combined["late_rate"])
        - (severe_weight * combined["severe_rate"])
        + (early_bonus_weight * combined["early_rate"])
    ).clip(0, 1)

    pbs = combined["PBS"]
    logger.info(f"PBS computed for {len(pbs)} applicants. "
                f"Mean={pbs.mean():.4f}, Std={pbs.std():.4f}")
    return pbs