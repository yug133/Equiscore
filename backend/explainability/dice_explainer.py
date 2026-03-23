import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import dice_ml
from dice_ml import Dice
from utils.logger import get_logger

logger = get_logger(__name__)

# Features the applicant can realistically change
# Excluded: immutable features like age, gender, region
ACTIONABLE_FEATURES = [
    "TRS",           # Transaction Regularity Score — improve by transacting regularly
    "PBS",           # Payment Behaviour Score — improve by paying on time
    "ISI",           # Income Stability Index — improve by stabilising income
    "DFS",           # Digital Footprint Score — improve by using digital payments
    "AMT_INCOME_TOTAL",   # Income — can change jobs
    "DAYS_EMPLOYED",      # Employment duration — stays longer
    "CNT_FAM_MEMBERS",    # Family size — minor factor
    "EXT_SOURCE_2",       # External score 2 — actionable over time
    "EXT_SOURCE_3",       # External score 3 — actionable over time
]


def build_dice_explainer(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    backend: str = "sklearn"
) -> Dice:
    """
    Build a DiCE counterfactual explainer for the trained model.

    DiCE (Diverse Counterfactual Explanations) generates minimum-change
    scenarios showing rejected applicants the exact behavioural changes
    needed to flip their prediction from rejected to approved.

    Only actionable features are varied in counterfactuals — immutable
    attributes like gender, region, and age are held fixed, ensuring
    the improvement recommendations are practically achievable.

    Args:
        model: Trained model instance (XGBClassifier or sklearn-compatible).
        X_train: Training feature DataFrame for DiCE data object.
        y_train: Training target Series for DiCE data object.
        backend: DiCE backend type. Use 'sklearn' for XGBoost. Default 'sklearn'.

    Returns:
        Configured Dice explainer instance.
    """
    logger.info("Building DiCE explainer...")

    # Combine features and target for DiCE data object
    train_data = X_train.copy()
    train_data["TARGET"] = y_train.values

    # Build DiCE data object
    dice_data = dice_ml.Data(
        dataframe=train_data,
        continuous_features=X_train.columns.tolist(),
        outcome_name="TARGET"
    )

    # Build DiCE model wrapper
    dice_model = dice_ml.Model(model=model, backend=backend)

    # Build DiCE explainer
    explainer = Dice(dice_data, dice_model, method="random")
    logger.info("DiCE explainer built successfully")
    return explainer


def generate_counterfactuals(
    explainer: Dice,
    X_applicant: pd.DataFrame,
    num_counterfactuals: int = 5,
    desired_class: int = 0,
    actionable_features: Optional[List[str]] = None
) -> Dict:
    """
    Generate counterfactual explanations for a rejected applicant.

    For a rejected applicant (predicted class=1, i.e. default risk),
    DiCE finds the minimum changes to their feature values that would
    flip the prediction to class=0 (approved/low risk).

    Each counterfactual represents one possible improvement path.
    Multiple diverse counterfactuals give the applicant choices —
    they can pick the path most suitable for their situation.

    Args:
        explainer: Fitted Dice explainer instance.
        X_applicant: Single-row DataFrame for the rejected applicant.
        num_counterfactuals: Number of alternative paths to generate.
                             Default 5 for diversity of options.
        desired_class: Target class to flip to. 0=approved. Default 0.
        actionable_features: Features that can be changed. If None,
                             uses ACTIONABLE_FEATURES constant.

    Returns:
        Dictionary with keys:
        - counterfactuals: List of counterfactual DataFrames
        - improvement_tips: List of human-readable improvement strings
        - validity_rate: Fraction of CFs that actually flip prediction
    """
    features = actionable_features or ACTIONABLE_FEATURES
    available = [f for f in features if f in X_applicant.columns]

    logger.info(f"Generating {num_counterfactuals} counterfactuals "
                f"using {len(available)} actionable features...")

    try:
        cf_result = explainer.generate_counterfactuals(
            X_applicant,
            total_CFs=num_counterfactuals,
            desired_class=desired_class,
            features_to_vary=available
        )

        cf_df = cf_result.cf_examples_list[0].final_cfs_df
        original = X_applicant.iloc[0]

        tips = []
        for _, cf_row in cf_df.iterrows():
            tip = {}
            for feat in available:
                if feat in cf_row and feat in original:
                    orig_val = original[feat]
                    cf_val = cf_row[feat]
                    if abs(cf_val - orig_val) > 0.001:
                        direction = "increase" if cf_val > orig_val else "decrease"
                        tip[feat] = {
                            "current": round(float(orig_val), 4),
                            "required": round(float(cf_val), 4),
                            "direction": direction,
                            "change": round(float(cf_val - orig_val), 4)
                        }
            tips.append(tip)

        logger.info(f"Generated {len(cf_df)} counterfactuals successfully")
        return {
            "counterfactuals": cf_df.to_dict("records"),
            "improvement_tips": tips,
            "validity_rate": len(cf_df) / num_counterfactuals
        }

    except Exception as e:
        logger.error(f"DiCE generation failed: {e}")
        return {
            "counterfactuals": [],
            "improvement_tips": [],
            "validity_rate": 0.0,
            "error": str(e)
        }


def format_tips_for_customer(
    improvement_tips: List[Dict],
    top_n: int = 3
) -> List[str]:
    """
    Format DiCE improvement tips into human-readable customer messages.

    Converts raw feature change dictionaries into plain English
    improvement suggestions that a customer can act on. Tips are
    ranked by the magnitude of change required — smallest changes
    first, as these are easiest for the applicant to achieve.

    Args:
        improvement_tips: List of improvement tip dicts from
                         generate_counterfactuals().
        top_n: Number of top tips to return. Default 3.

    Returns:
        List of human-readable improvement suggestion strings.
    """
    feature_messages = {
        "TRS": "Make transactions more regularly every month",
        "PBS": "Pay all instalments on time or early",
        "ISI": "Maintain a stable and consistent income source",
        "DFS": "Use digital payment methods (UPI, cards) more frequently",
        "AMT_INCOME_TOTAL": "Increase your monthly income",
        "DAYS_EMPLOYED": "Stay in your current employment longer",
        "EXT_SOURCE_2": "Build your external credit profile over time",
        "EXT_SOURCE_3": "Improve your external creditworthiness score",
        "CNT_FAM_MEMBERS": "Update your family size information",
    }

    messages = []
    for tip in improvement_tips[:top_n]:
        for feat, change in tip.items():
            base_msg = feature_messages.get(feat, f"Improve {feat}")
            direction = change["direction"]
            current = change["current"]
            required = change["required"]
            msg = (f"{base_msg}. "
                   f"Current: {current:.3f} → Required: {required:.3f} "
                   f"({direction})")
            messages.append(msg)
            break  # One tip per counterfactual

    logger.info(f"Formatted {len(messages)} customer tips")
    return messages