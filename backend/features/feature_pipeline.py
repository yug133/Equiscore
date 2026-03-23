import pandas as pd
import numpy as np
from typing import Dict, Tuple
from utils.logger import get_logger

from features.transaction_regularity import compute_transaction_regularity
from features.payment_behaviour import compute_payment_behaviour_score
from features.income_stability import compute_income_stability_index
from features.digital_footprint import compute_digital_footprint_score
from features.geo_income_index import compute_geo_income_index

logger = get_logger(__name__)

# Feature weights for final combined feature vector
FEATURE_WEIGHTS = {
    "TRS": 0.20,  # Transaction Regularity Score
    "PBS": 0.25,  # Payment Behaviour Score
    "ISI": 0.20,  # Income Stability Index
    "DFS": 0.15,  # Digital Footprint Score
    "GII": 0.20,  # Geo Income Index
}


def compute_all_features(
    tables: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Compute all 5 custom features and return as a single DataFrame.

    Runs each feature computation pipeline and joins results on
    SK_ID_CURR. Missing values for applicants not present in
    auxiliary tables are filled with neutral score of 0.5.

    Args:
        tables: Dictionary of DataFrames as returned by load_all_tables().
                Required keys: application_train, installments,
                pos_cash, credit_card, bureau.

    Returns:
        DataFrame indexed by SK_ID_CURR with columns:
        TRS, PBS, ISI, DFS, GII — all scaled 0 to 1.
    """
    logger.info("Computing TRS — Transaction Regularity Score...")
    trs = compute_transaction_regularity(tables["installments"])

    logger.info("Computing PBS — Payment Behaviour Score...")
    pbs = compute_payment_behaviour_score(tables["installments"])

    logger.info("Computing ISI — Income Stability Index...")
    isi = compute_income_stability_index(
        tables["pos_cash"],
        tables["credit_card"]
    )

    logger.info("Computing DFS — Digital Footprint Score...")
    dfs = compute_digital_footprint_score(
        tables["credit_card"],
        tables["pos_cash"],
        tables["bureau"]
    )

    logger.info("Computing GII — Geo Income Index...")
    gii = compute_geo_income_index(tables["application_train"])

    # Combine all features into one DataFrame
    feature_df = pd.DataFrame({
        "TRS": trs,
        "PBS": pbs,
        "ISI": isi,
        "DFS": dfs,
        "GII": gii,
    }).fillna(0.5)  # Neutral score for missing applicants

    logger.info(f"All features computed. Shape: {feature_df.shape}")
    logger.info(f"Feature summary:\n{feature_df.describe().round(4)}")
    return feature_df


def merge_features_with_application(
    application_df: pd.DataFrame,
    feature_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge custom features with the preprocessed application table.

    Left joins feature_df onto application_df on SK_ID_CURR so that
    every applicant row gets their 5 feature scores appended.
    Missing feature scores are filled with 0.5 (neutral).

    Args:
        application_df: Preprocessed application DataFrame.
                        Must have SK_ID_CURR as a column or index.
        feature_df: DataFrame with TRS, PBS, ISI, DFS, GII columns,
                    indexed by SK_ID_CURR.

    Returns:
        Merged DataFrame with all original columns plus 5 new
        feature columns.
    """
    if application_df.index.name == "SK_ID_CURR":
        application_df = application_df.reset_index()

    merged = application_df.merge(
        feature_df.reset_index(),
        on="SK_ID_CURR",
        how="left"
    )

    for col in ["TRS", "PBS", "ISI", "DFS", "GII"]:
        merged[col] = merged[col].fillna(0.5)

    logger.info(f"Merged DataFrame shape: {merged.shape}")
    return merged


def compute_composite_behaviour_score(
    feature_df: pd.DataFrame,
    weights: Dict[str, float] = FEATURE_WEIGHTS
) -> pd.Series:
    """
    Compute a single composite behaviour score from all 5 features.

    Weighted average of TRS, PBS, ISI, DFS, GII using predefined
    weights. This composite score is used as an additional input
    feature to the ML models alongside individual feature scores.

    Formula:
        CBS = 0.20*TRS + 0.25*PBS + 0.20*ISI + 0.15*DFS + 0.20*GII

    Args:
        feature_df: DataFrame with columns TRS, PBS, ISI, DFS, GII
                    indexed by SK_ID_CURR.
        weights: Dictionary mapping feature name to weight.
                 Weights must sum to 1.0.

    Returns:
        Series indexed by SK_ID_CURR with composite score 0 to 1.
    """
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
        logger.warning(f"Feature weights sum to {total_weight}, expected 1.0")

    cbs = sum(
        feature_df[feat] * weight
        for feat, weight in weights.items()
        if feat in feature_df.columns
    )
    cbs = cbs.clip(0, 1).rename("CBS")

    logger.info(f"Composite Behaviour Score computed. "
                f"Mean={cbs.mean():.4f}, Std={cbs.std():.4f}")
    return cbs


def get_feature_summary(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return descriptive statistics for all computed features.

    Useful for validating feature distributions before model training.
    Checks for unexpected all-zero or all-one features that may
    indicate a bug in the feature computation pipeline.

    Args:
        feature_df: DataFrame with TRS, PBS, ISI, DFS, GII columns.

    Returns:
        Summary DataFrame with mean, std, min, max per feature.
    """
    summary = feature_df[["TRS", "PBS", "ISI", "DFS", "GII"]].describe().T
    summary["zero_rate"] = (feature_df == 0).mean()
    summary["one_rate"] = (feature_df == 1).mean()

    logger.info(f"Feature summary:\n{summary.round(4)}")
    return summary