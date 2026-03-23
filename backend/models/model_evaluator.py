import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    brier_score_loss
)
from utils.logger import get_logger

logger = get_logger(__name__)


def compute_auc_roc(
    y_true: pd.Series,
    y_proba: np.ndarray
) -> float:
    """
    Compute Area Under the ROC Curve (AUC-ROC).

    AUC-ROC measures the model's ability to distinguish between
    defaulters and non-defaulters. Higher is better.
    Target for EquiScore: AUC > 0.75.

    Args:
        y_true: True binary labels (0=repaid, 1=default).
        y_proba: Predicted default probabilities from model.

    Returns:
        AUC-ROC score between 0 and 1.
    """
    auc = roc_auc_score(y_true, y_proba)
    logger.info(f"AUC-ROC: {auc:.4f}")
    return auc


def compute_ks_statistic(
    y_true: pd.Series,
    y_proba: np.ndarray
) -> float:
    """
    Compute Kolmogorov-Smirnov (KS) statistic.

    KS measures the maximum separation between the cumulative
    distribution of predicted scores for defaulters vs non-defaulters.
    Widely used in credit scoring to measure model discrimination.
    Higher KS = better separation between good and bad applicants.
    Industry benchmark: KS > 0.30 is considered good.

    Args:
        y_true: True binary labels (0=repaid, 1=default).
        y_proba: Predicted default probabilities from model.

    Returns:
        KS statistic between 0 and 1.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ks = float(np.max(tpr - fpr))
    logger.info(f"KS Statistic: {ks:.4f}")
    return ks


def compute_gini_coefficient(
    y_true: pd.Series,
    y_proba: np.ndarray
) -> float:
    """
    Compute Gini coefficient from AUC-ROC.

    Gini = 2 * AUC - 1
    Normalizes AUC to a -1 to 1 scale where 0 = random model
    and 1 = perfect model. Standard metric in credit risk industry.

    Args:
        y_true: True binary labels (0=repaid, 1=default).
        y_proba: Predicted default probabilities from model.

    Returns:
        Gini coefficient between -1 and 1.
    """
    auc = compute_auc_roc(y_true, y_proba)
    gini = 2 * auc - 1
    logger.info(f"Gini Coefficient: {gini:.4f}")
    return gini


def compute_demographic_parity_gap(
    y_proba: np.ndarray,
    sensitive_features: pd.Series
) -> Dict[str, float]:
    """
    Compute Demographic Parity Gap (DPG) across sensitive groups.

    DPG measures the difference in positive prediction rates between
    demographic groups. A gap of 0 means equal approval rates.
    Larger gaps indicate potential discriminatory outcomes.

    Formula per group pair:
        DPG = |P(score > threshold | group A) -
               P(score > threshold | group B)|

    Args:
        y_proba: Predicted default probabilities from model.
        sensitive_features: Series of group labels per applicant.

    Returns:
        Dictionary mapping group pairs to their DPG values.
    """
    threshold = 0.5
    predictions = (y_proba < threshold).astype(int)  # 1=approved

    results = {}
    groups = sensitive_features.unique()
    approval_rates = {}

    for group in groups:
        mask = sensitive_features == group
        approval_rates[group] = predictions[mask].mean()
        logger.info(f"Group '{group}' approval rate: "
                    f"{approval_rates[group]:.4f}")

    # Compute pairwise gaps
    group_list = list(approval_rates.keys())
    for i in range(len(group_list)):
        for j in range(i + 1, len(group_list)):
            g1, g2 = group_list[i], group_list[j]
            gap = abs(approval_rates[g1] - approval_rates[g2])
            results[f"{g1}_vs_{g2}"] = round(gap, 4)

    logger.info(f"Demographic Parity Gaps: {results}")
    return results


def compute_all_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    sensitive_features: Optional[pd.Series] = None,
    model_name: str = "model"
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for a trained model.

    Runs AUC-ROC, KS statistic, Gini coefficient, Brier score,
    and optionally Demographic Parity Gap if sensitive features
    are provided. Returns all metrics as a single dictionary
    for easy comparison across models.

    Args:
        y_true: True binary labels (0=repaid, 1=default).
        y_proba: Predicted default probabilities from model.
        sensitive_features: Optional Series of group labels for
                            fairness metrics. If None, DPG is skipped.
        model_name: Name of the model for logging. Default 'model'.

    Returns:
        Dictionary with keys: model, auc_roc, ks_statistic,
        gini, brier_score, dpg (if sensitive_features provided).
    """
    logger.info(f"Evaluating {model_name}...")

    metrics = {
        "model": model_name,
        "auc_roc": compute_auc_roc(y_true, y_proba),
        "ks_statistic": compute_ks_statistic(y_true, y_proba),
        "gini": compute_gini_coefficient(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
    }

    if sensitive_features is not None:
        metrics["dpg"] = compute_demographic_parity_gap(
            y_proba, sensitive_features
        )

    logger.info(f"{model_name} metrics: {metrics}")
    return metrics


def compare_models(
    results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Compare evaluation metrics across multiple trained models.

    Takes a dictionary of model results and returns a formatted
    DataFrame for easy side-by-side comparison of AUC, KS, Gini
    and Brier score across logistic, RF, XGBoost and fair XGBoost.

    Args:
        results: Dictionary mapping model name to its metrics dict.
                 e.g. {"logistic": {...}, "xgboost": {...}}

    Returns:
        DataFrame with one row per model and metric columns.
        Sorted by AUC-ROC descending.
    """
    rows = []
    for name, metrics in results.items():
        rows.append({
            "model": name,
            "auc_roc": metrics.get("auc_roc", None),
            "ks_statistic": metrics.get("ks_statistic", None),
            "gini": metrics.get("gini", None),
            "brier_score": metrics.get("brier_score", None),
        })

    comparison = (
        pd.DataFrame(rows)
        .sort_values("auc_roc", ascending=False)
        .reset_index(drop=True)
    )

    logger.info(f"Model comparison:\n{comparison}")
    return comparison