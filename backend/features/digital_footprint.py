import pandas as pd
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)

# Weights for each digital footprint component
CREDIT_CARD_ACTIVITY_WEIGHT = 0.35
POS_ACTIVITY_WEIGHT = 0.35
BUREAU_ENQUIRY_WEIGHT = 0.30

# Minimum activity threshold to avoid zero scores
MIN_ACTIVITY_CLIP = 1e-6


def compute_credit_card_activity_score(
    credit_card_df: pd.DataFrame
) -> pd.Series:
    """
    Compute digital activity score from credit card usage patterns.

    Measures how actively an applicant uses digital credit products.
    Uses AMT_DRAWINGS_CURRENT (total drawings) and
    CNT_DRAWINGS_CURRENT (number of drawings) as activity proxies.

    Higher usage with consistent repayment indicates digital
    financial engagement — a positive creditworthiness signal.

    Args:
        credit_card_df: credit_card_balance table from Home Credit.
                        Must contain SK_ID_CURR, AMT_DRAWINGS_CURRENT,
                        CNT_DRAWINGS_CURRENT.

    Returns:
        Series indexed by SK_ID_CURR with activity score 0 to 1.
    """
    df = credit_card_df.copy()

    activity = (
        df.groupby("SK_ID_CURR")
        .agg(
            total_drawings=("AMT_DRAWINGS_CURRENT", "sum"),
            total_drawing_count=("CNT_DRAWINGS_CURRENT", "sum")
        )
        .reset_index()
    )

    # Normalize each metric to 0-1
    for col in ["total_drawings", "total_drawing_count"]:
        max_val = activity[col].max()
        if max_val > 0:
            activity[col] = activity[col] / max_val
        else:
            activity[col] = 0.0

    activity["cc_score"] = (
        activity["total_drawings"] * 0.5 +
        activity["total_drawing_count"] * 0.5
    ).clip(MIN_ACTIVITY_CLIP, 1.0)

    cc_score = activity.set_index("SK_ID_CURR")["cc_score"]
    logger.info(f"Credit card activity score computed for {len(cc_score)} applicants")
    return cc_score


def compute_pos_activity_score(
    pos_cash_df: pd.DataFrame
) -> pd.Series:
    """
    Compute digital activity score from POS cash loan usage.

    Measures breadth of POS loan engagement — number of active
    loans and total instalments paid. More POS activity indicates
    greater participation in formal digital financial systems.

    Args:
        pos_cash_df: POS_CASH_balance table from Home Credit.
                     Must contain SK_ID_CURR, CNT_INSTALMENT,
                     NAME_CONTRACT_STATUS.

    Returns:
        Series indexed by SK_ID_CURR with POS activity score 0 to 1.
    """
    df = pos_cash_df.copy()

    activity = (
        df.groupby("SK_ID_CURR")
        .agg(
            total_instalments=("CNT_INSTALMENT", "sum"),
            active_months=("MONTHS_BALANCE", "count")
        )
        .reset_index()
    )

    for col in ["total_instalments", "active_months"]:
        max_val = activity[col].max()
        if max_val > 0:
            activity[col] = activity[col] / max_val
        else:
            activity[col] = 0.0

    activity["pos_score"] = (
        activity["total_instalments"] * 0.5 +
        activity["active_months"] * 0.5
    ).clip(MIN_ACTIVITY_CLIP, 1.0)

    pos_score = activity.set_index("SK_ID_CURR")["pos_score"]
    logger.info(f"POS activity score computed for {len(pos_score)} applicants")
    return pos_score


def compute_bureau_enquiry_score(
    bureau_df: pd.DataFrame
) -> pd.Series:
    """
    Compute digital footprint score from credit bureau enquiries.

    More bureau records indicate wider engagement with formal
    financial institutions. However, too many recent enquiries
    can signal credit-seeking stress — so we use a log-scaled score
    to reward breadth without penalising high enquiry counts too much.

    Args:
        bureau_df: Bureau table from Home Credit.
                   Must contain SK_ID_CURR, DAYS_CREDIT.

    Returns:
        Series indexed by SK_ID_CURR with bureau score 0 to 1.
    """
    df = bureau_df.copy()

    enquiry_counts = (
        df.groupby("SK_ID_CURR")
        .size()
        .rename("enquiry_count")
    )

    # Log scale to reduce impact of very high enquiry counts
    log_scaled = np.log1p(enquiry_counts)
    bureau_score = (log_scaled / log_scaled.max()).clip(0, 1)
    bureau_score.name = "bureau_score"

    logger.info(f"Bureau enquiry score computed for {len(bureau_score)} applicants")
    return bureau_score


def compute_digital_footprint_score(
    credit_card_df: pd.DataFrame,
    pos_cash_df: pd.DataFrame,
    bureau_df: pd.DataFrame
) -> pd.Series:
    """
    Compute the Digital Footprint Score (DFS) for each applicant.

    Combines credit card activity, POS loan activity, and bureau
    enquiry breadth into a single weighted score between 0 and 1.

    Formula:
        DFS = (0.35 * cc_score)
            + (0.35 * pos_score)
            + (0.30 * bureau_score)

    Applicants with no digital footprint receive a score of 0.
    Missing component scores are filled with 0 before combining.

    Args:
        credit_card_df: credit_card_balance table from Home Credit.
        pos_cash_df: POS_CASH_balance table from Home Credit.
        bureau_df: Bureau table from Home Credit.

    Returns:
        Series indexed by SK_ID_CURR with DFS score between 0 and 1.
    """
    cc_score = compute_credit_card_activity_score(credit_card_df)
    pos_score = compute_pos_activity_score(pos_cash_df)
    bureau_score = compute_bureau_enquiry_score(bureau_df)

    combined = pd.DataFrame({
        "cc_score": cc_score,
        "pos_score": pos_score,
        "bureau_score": bureau_score
    }).fillna(0)

    combined["DFS"] = (
        CREDIT_CARD_ACTIVITY_WEIGHT * combined["cc_score"] +
        POS_ACTIVITY_WEIGHT * combined["pos_score"] +
        BUREAU_ENQUIRY_WEIGHT * combined["bureau_score"]
    ).clip(0, 1)

    dfs = combined["DFS"]
    logger.info(f"DFS computed for {len(dfs)} applicants. "
                f"Mean={dfs.mean():.4f}, Std={dfs.std():.4f}")
    return dfs