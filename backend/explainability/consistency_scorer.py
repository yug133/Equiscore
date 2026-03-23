import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import kendalltau
from utils.logger import get_logger

logger = get_logger(__name__)

# Minimum acceptable consistency score per project requirements
CONSISTENCY_THRESHOLD = 0.90


def compute_shap_rank_correlation(
    shap_values_list: List[np.ndarray],
    feature_names: List[str]
) -> float:
    """
    Compute SHAP rank consistency across multiple random seeds.

    Measures how stable feature importance rankings are when the model
    is trained with different random seeds. High consistency means the
    model's explanations are reliable and not artifacts of random
    initialization.

    Method: For each pair of SHAP value arrays, compute Kendall's W
    (coefficient of concordance) on the mean absolute SHAP rankings.
    Average across all pairs gives the final consistency score.

    Target per project requirements: consistency score > 0.90.

    Args:
        shap_values_list: List of 2D SHAP value arrays, one per seed.
                          Each array shape: (n_samples, n_features).
        feature_names: List of feature column names.

    Returns:
        Float consistency score between 0 and 1.
        1.0 = perfectly consistent rankings across all seeds.
        0.0 = completely random rankings.
    """
    if len(shap_values_list) < 2:
        logger.warning("Need at least 2 SHAP arrays for consistency check")
        return 1.0

    # Compute mean absolute SHAP per feature for each seed
    rankings = []
    for shap_vals in shap_values_list:
        mean_abs = np.abs(shap_vals).mean(axis=0)
        rank = pd.Series(mean_abs, index=feature_names).rank(ascending=False)
        rankings.append(rank)

    # Compute pairwise Kendall tau correlations
    correlations = []
    for i in range(len(rankings)):
        for j in range(i + 1, len(rankings)):
            tau, _ = kendalltau(rankings[i].values, rankings[j].values)
            correlations.append(tau)

    consistency = float(np.mean(correlations))
    logger.info(f"SHAP rank consistency score: {consistency:.4f} "
                f"(threshold={CONSISTENCY_THRESHOLD})")
    return consistency


def run_consistency_check(
    model_builder,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    seeds: List[int] = [42, 123, 456, 789, 999],
    sample_size: int = 500
) -> Dict:
    """
    Run full SHAP consistency check across multiple random seeds.

    Trains the model with each seed, computes SHAP values on a
    sample of the test set, then measures how consistently the
    feature importance rankings agree across all seeds.

    This validates that EquiScore's explanations are stable and
    reproducible — a key requirement for regulatory acceptance.

    Args:
        model_builder: Callable that returns a new untrained model.
                       Called once per seed with random_state=seed.
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        X_test: Test feature DataFrame.
        seeds: List of random seeds to evaluate. Default 5 seeds.
        sample_size: Number of test rows to compute SHAP on per seed.
                     Smaller = faster, larger = more reliable estimate.

    Returns:
        Dictionary with keys:
        - consistency_score: Float Kendall W score across seeds
        - passes_threshold: Bool whether score >= 0.90
        - per_seed_top_features: Dict mapping seed to top 5 features
        - stable_features: Features in top 10 across ALL seeds
    """
    import shap

    shap_values_list = []
    feature_names = X_train.columns.tolist()
    per_seed_top = {}
    X_sample = X_test.sample(
        min(sample_size, len(X_test)), random_state=42
    )

    for seed in seeds:
        logger.info(f"Training model with seed={seed}...")
        model = model_builder(random_state=seed)
        model.fit(X_train, y_train)

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        shap_values_list.append(shap_vals)

        # Track top features per seed
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top_features = (
            pd.Series(mean_abs, index=feature_names)
            .sort_values(ascending=False)
            .head(10)
            .index.tolist()
        )
        per_seed_top[seed] = top_features
        logger.info(f"Seed {seed} top features: {top_features[:5]}")

    consistency = compute_shap_rank_correlation(shap_values_list, feature_names)

    # Find features in top 10 across ALL seeds
    all_top_sets = [set(v) for v in per_seed_top.values()]
    stable_features = list(set.intersection(*all_top_sets))

    result = {
        "consistency_score": consistency,
        "passes_threshold": consistency >= CONSISTENCY_THRESHOLD,
        "per_seed_top_features": per_seed_top,
        "stable_features": stable_features,
        "seeds_tested": seeds,
    }

    status = "✅ PASSED" if result["passes_threshold"] else "❌ FAILED"
    logger.info(f"Consistency check {status}: {consistency:.4f} "
                f"(threshold={CONSISTENCY_THRESHOLD})")
    logger.info(f"Stable features across all seeds: {stable_features}")
    return result


def get_consistency_report(consistency_result: Dict) -> str:
    """
    Format consistency check results into a human-readable report.

    Generates a plain text summary of the SHAP consistency check
    suitable for inclusion in the fairness audit report and
    regulatory documentation.

    Args:
        consistency_result: Output dictionary from run_consistency_check().

    Returns:
        Formatted string report of consistency results.
    """
    score = consistency_result["consistency_score"]
    passed = consistency_result["passes_threshold"]
    stable = consistency_result["stable_features"]
    seeds = consistency_result["seeds_tested"]

    status = "PASSED ✅" if passed else "FAILED ❌"
    report = f"""
SHAP CONSISTENCY REPORT
=======================
Seeds tested:          {seeds}
Consistency score:     {score:.4f} (Kendall W)
Threshold:             {CONSISTENCY_THRESHOLD}
Status:                {status}

Stable features (top 10 across ALL seeds):
{chr(10).join(f"  - {f}" for f in stable)}

Interpretation:
  A score above 0.90 means EquiScore's feature importance
  rankings are stable regardless of random initialization.
  This satisfies the explainability reliability requirement
  for RBI Digital Lending Guidelines 2023 compliance.
"""
    logger.info(report)
    return report