import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

# Minimum months of data required for a reliable ISI score
MIN_MONTHS_REQUIRED = 3


def compute_monthly_income_estimates(
    pos_cash_df: pd.DataFrame,
    credit_card_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Estimate monthly income proxy from POS cash and credit card balances.

    Since the Home Credit dataset does not contain direct monthly income
    records, we use CNT_INSTALMENT and AMT_BALANCE as proxies for
    monthly financial activity, aggregated per applicant per month.

    Args:
        pos_cash_df: POS_CASH_balance table from Home Credit.
                     Must contain SK_ID_CURR, MONTHS_BALANCE,
                     CNT_INSTALMENT.
        credit_card_df: credit_card_balance table from Home Credit.
                        Must contain SK_ID_CURR, MONTHS_BALANCE,
                        AMT_BALANCE.

    Returns:
        DataFrame with columns: SK_ID_CURR, month, income_proxy.
    """
    # POS cash proxy: instalment count per month
    pos = pos_cash_df[["SK_ID_CURR", "MONTHS_BALANCE", "CNT_INSTALMENT"]].copy()
    pos = pos.rename(columns={
        "MONTHS_BALANCE": "month",
        "CNT_INSTALMENT": "income_proxy"
    })

    # Credit card proxy: balance per month
    cc = credit_card_df[["SK_ID_CURR", "MONTHS_BALANCE", "AMT_BALANCE"]].copy()
    cc = cc.rename(columns={
        "MONTHS_BALANCE": "month",
        "AMT_BALANCE": "income_proxy"
    })

    combined = pd.concat([pos, cc], ignore_index=True)

    monthly = (
        combined.groupby(["SK_ID_CURR", "month"])["income_proxy"]
        .sum()
        .reset_index()
    )

    logger.info(f"Monthly income proxies computed for "
                f"{monthly['SK_ID_CURR'].nunique()} applicants")
    return monthly


def compute_income_coefficient_of_variation(
    monthly_income_df: pd.DataFrame
) -> pd.Series:
    """
    Compute the coefficient of variation (CV) of monthly income per applicant.

    CV = std / mean of monthly income proxy values.
    Lower CV = more stable income = better creditworthiness signal.

    Args:
        monthly_income_df: DataFrame with SK_ID_CURR, month, income_proxy.
                           Output of compute_monthly_income_estimates.

    Returns:
        Series indexed by SK_ID_CURR with CV values.
        Returns NaN for applicants with fewer than MIN_MONTHS_REQUIRED months.
    """
    def cv(series: pd.Series) -> float:
        if len(series) < MIN_MONTHS_REQUIRED:
            return np.nan
        mean_val = series.mean()
        if mean_val == 0:
            return np.nan
        return series.std() / mean_val

    cv_series = (
        monthly_income_df
        .groupby("SK_ID_CURR")["income_proxy"]
        .apply(cv)
        .rename("income_cv")
    )

    logger.info(f"Income CV computed. "
                f"NaN count: {cv_series.isna().sum()}")
    return cv_series


def compute_income_stability_index(
    pos_cash_df: pd.DataFrame,
    credit_card_df: pd.DataFrame
) -> pd.Series:
    """
    Compute the Income Stability Index (ISI) for each applicant.

    Transforms the coefficient of variation into a 0-1 score where:
        1 = perfectly stable income
        0 = highly volatile income

    Formula:
        ISI = 1 / (1 + CV)

    Applicants with insufficient history receive a neutral score of 0.5.

    Args:
        pos_cash_df: POS_CASH_balance table from Home Credit.
        credit_card_df: credit_card_balance table from Home Credit.

    Returns:
        Series indexed by SK_ID_CURR with ISI score between 0 and 1.
    """
    monthly_income = compute_monthly_income_estimates(pos_cash_df, credit_card_df)
    cv_series = compute_income_coefficient_of_variation(monthly_income)

    # Transform CV to stability score
    isi = (1 / (1 + cv_series)).fillna(0.5).rename("ISI")
    isi = isi.clip(0, 1)

    logger.info(f"ISI computed for {len(isi)} applicants. "
                f"Mean={isi.mean():.4f}, Std={isi.std():.4f}")
    return isi


def flag_unstable_income(
    isi: pd.Series,
    threshold: float = 0.4
) -> pd.Series:
    """
    Flag applicants with unstable income based on ISI score.

    Applicants below the threshold are considered to have
    volatile income patterns, which increases default risk.

    Args:
        isi: Series of ISI scores indexed by SK_ID_CURR.
        threshold: Score below which applicant is flagged. Default 0.4.

    Returns:
        Boolean Series indexed by SK_ID_CURR.
        True = unstable income (flagged), False = stable.
    """
    flagged = isi < threshold
    logger.info(f"Flagged {flagged.sum()} applicants with unstable income "
                f"(threshold={threshold})")
    return flagged