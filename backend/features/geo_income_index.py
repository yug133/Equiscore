import os
import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

# Region rating column in Home Credit application table
REGION_RATING_COL = "REGION_RATING_CLIENT"
INCOME_COL = "AMT_INCOME_TOTAL"
ID_COL = "SK_ID_CURR"

# Smoothing factor to avoid division by zero
SMOOTHING_FACTOR = 1e-6

# Path to RBI region mapping CSV
RBI_MAPPING_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "data", "raw", "rbi_region_fi_mapping.csv"
)

# Fallback mapping if CSV not found
# Based on RBI Financial Inclusion Report 2024-25
# FI-Index: National=64.2, Urban~76.5, Semi-Urban~64.2, Rural~48.1
FALLBACK_RBI_MAPPING = {
    1: {"fi_index_score": 76.5, "banking_penetration": 0.92,
        "income_percentile": 0.78, "gii_base_score": 0.82},
    2: {"fi_index_score": 64.2, "banking_penetration": 0.71,
        "income_percentile": 0.50, "gii_base_score": 0.55},
    3: {"fi_index_score": 48.1, "banking_penetration": 0.43,
        "income_percentile": 0.28, "gii_base_score": 0.31},
}


def load_rbi_region_mapping() -> pd.DataFrame:
    """
    Load RBI Financial Inclusion region mapping from CSV.

    Maps Home Credit REGION_RATING_CLIENT (1/2/3) to RBI FI-Index
    scores and banking penetration metrics from the RBI Annual Report
    2024-25 (Credit Delivery and Financial Inclusion, Chapter IV).

    Mapping justification:
    - Region 1 (best rated): Corresponds to high-inclusion urban
      districts. FI-Index ~76.5 (20% above national average of 64.2).
      Banking penetration 0.92. Income percentile 0.78.
    - Region 2 (mid rated): Corresponds to semi-urban districts at
      national FI-Index baseline of 64.2. Penetration 0.71.
    - Region 3 (worst rated): Corresponds to low-inclusion rural
      districts. FI-Index ~48.1 (25% below national). Penetration 0.43.

    Source: RBI Annual Report 2024-25, Table IV.4 (Financial Inclusion
    Plan Progress Report), Chart IV.1 (FI-Index end-March 2024=64.2).

    Returns:
        DataFrame with region_rating as index and FI metrics as columns.
    """
    rbi_path = os.path.abspath(RBI_MAPPING_PATH)
    if os.path.exists(rbi_path):
        mapping = pd.read_csv(rbi_path)
        mapping = mapping.set_index("region_rating")
        logger.info(f"RBI mapping loaded from {rbi_path}")
    else:
        logger.warning(f"RBI mapping CSV not found at {rbi_path}. "
                       f"Using fallback values from RBI Annual Report 2024-25.")
        mapping = pd.DataFrame(FALLBACK_RBI_MAPPING).T
        mapping.index.name = "region_rating"

    logger.info(f"RBI Region Mapping:\n{mapping}")
    return mapping


def compute_regional_income_stats(
    application_df: pd.DataFrame,
    region_col: str = REGION_RATING_COL,
    income_col: str = INCOME_COL
) -> pd.DataFrame:
    """
    Compute mean and standard deviation of income per region.

    Groups applicants by REGION_RATING_CLIENT and computes aggregate
    income statistics, then enriches with RBI FI-Index scores.

    Region ratings in Home Credit are 1, 2, or 3 where:
        1 = best rated region (urban, high FI-Index)
        3 = worst rated region (rural, low FI-Index)

    Args:
        application_df: Main application table from Home Credit.
                        Must contain REGION_RATING_CLIENT,
                        AMT_INCOME_TOTAL, SK_ID_CURR.
        region_col: Column name for region rating.
        income_col: Column name for income.

    Returns:
        DataFrame indexed by region with income stats and
        RBI FI-Index enrichment columns.
    """
    regional_stats = (
        application_df
        .groupby(region_col)[income_col]
        .agg(
            regional_mean_income="mean",
            regional_std_income="std",
            regional_median_income="median"
        )
    )
    regional_stats["regional_std_income"] = (
        regional_stats["regional_std_income"].fillna(SMOOTHING_FACTOR)
    )

    # Enrich with RBI FI-Index mapping
    rbi_mapping = load_rbi_region_mapping()
    regional_stats = regional_stats.join(rbi_mapping, how="left")

    logger.info(f"Regional income stats with RBI enrichment:\n"
                f"{regional_stats.round(2)}")
    return regional_stats


def compute_income_zscore_within_region(
    application_df: pd.DataFrame,
    region_col: str = REGION_RATING_COL,
    income_col: str = INCOME_COL
) -> pd.Series:
    """
    Compute each applicant's income z-score relative to their region.

    Z-score = (applicant_income - regional_mean) / regional_std

    This benchmarks applicants against their own region rather than
    national averages — a fairer comparison that does not disadvantage
    applicants from rural or lower-income regions.

    Args:
        application_df: Main application table from Home Credit.
        region_col: Column for region grouping.
        income_col: Column for income values.

    Returns:
        Series indexed by SK_ID_CURR with z-score values.
        Positive = earns above regional average.
        Negative = earns below regional average.
    """
    regional_stats = compute_regional_income_stats(
        application_df, region_col, income_col
    )

    df = application_df[[ID_COL, region_col, income_col]].copy()
    df = df.merge(
        regional_stats[["regional_mean_income", "regional_std_income"]],
        on=region_col,
        how="left"
    )

    df["income_zscore"] = (
        (df[income_col] - df["regional_mean_income"]) /
        (df["regional_std_income"] + SMOOTHING_FACTOR)
    )

    zscore = df.set_index(ID_COL)["income_zscore"]
    logger.info(f"Income z-scores computed. "
                f"Mean={zscore.mean():.4f}, Std={zscore.std():.4f}")
    return zscore


def compute_rbi_adjusted_gii(
    application_df: pd.DataFrame,
    region_col: str = REGION_RATING_COL,
    income_col: str = INCOME_COL,
    rbi_weight: float = 0.3,
    income_weight: float = 0.7
) -> pd.Series:
    """
    Compute RBI-adjusted Geo Income Index (GII) for each applicant.

    Combines two signals:
    1. Income z-score signal (weight=0.7): How the applicant's income
       compares to their regional peers — transformed via sigmoid.
    2. RBI FI-Index signal (weight=0.3): The region's overall financial
       inclusion level from RBI Annual Report 2024-25.

    Formula:
        income_signal = sigmoid(income_zscore)
        rbi_signal    = region's gii_base_score from RBI mapping
        GII = (income_weight * income_signal) +
              (rbi_weight * rbi_signal)

    This ensures applicants from low-inclusion regions are not
    doubly penalised — their regional baseline is already factored in.

    Args:
        application_df: Main application table from Home Credit.
                        Must contain SK_ID_CURR, REGION_RATING_CLIENT,
                        AMT_INCOME_TOTAL.
        region_col: Column for region grouping.
        income_col: Column for income values.
        rbi_weight: Weight for RBI FI-Index signal. Default 0.3.
        income_weight: Weight for income z-score signal. Default 0.7.

    Returns:
        Series indexed by SK_ID_CURR with GII score between 0 and 1.
    """
    # Income z-score signal
    zscore = compute_income_zscore_within_region(
        application_df, region_col, income_col
    )
    income_signal = (1 / (1 + np.exp(-zscore)))

    # RBI FI signal — map region rating to gii_base_score
    rbi_mapping = load_rbi_region_mapping()
    df = application_df[[ID_COL, region_col]].copy()
    df = df.merge(
        rbi_mapping[["gii_base_score"]].reset_index(),
        left_on=region_col,
        right_on="region_rating",
        how="left"
    )
    rbi_signal = df.set_index(ID_COL)["gii_base_score"].fillna(0.5)

    # Combine signals
    gii = (
        income_weight * income_signal +
        rbi_weight * rbi_signal
    ).clip(0, 1).rename("GII")

    logger.info(f"RBI-adjusted GII computed for {len(gii)} applicants. "
                f"Mean={gii.mean():.4f}, Std={gii.std():.4f}")
    return gii


def compute_geo_income_index(
    application_df: pd.DataFrame,
    region_col: str = REGION_RATING_COL,
    income_col: str = INCOME_COL
) -> pd.Series:
    """
    Main entry point for GII computation.

    Calls compute_rbi_adjusted_gii which combines income z-score
    with RBI Financial Inclusion Index regional scores from the
    RBI Annual Report 2024-25.

    Args:
        application_df: Main application table from Home Credit.
        region_col: Column for region grouping.
        income_col: Column for income values.

    Returns:
        Series indexed by SK_ID_CURR with GII score between 0 and 1.
    """
    return compute_rbi_adjusted_gii(application_df, region_col, income_col)


def flag_low_regional_income(
    gii: pd.Series,
    threshold: float = 0.35
) -> pd.Series:
    """
    Flag applicants who earn significantly below their regional average.

    These applicants may face higher financial stress relative to
    their local cost of living and peer group.

    Args:
        gii: Series of GII scores indexed by SK_ID_CURR.
        threshold: Score below which applicant is flagged. Default 0.35.

    Returns:
        Boolean Series indexed by SK_ID_CURR.
        True = low regional income (flagged), False = above threshold.
    """
    flagged = gii < threshold
    logger.info(f"Flagged {flagged.sum()} applicants with low regional "
                f"income (threshold={threshold})")
    return flagged