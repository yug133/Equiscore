"""
API Dependencies Module
Shared model, explainer, and database instances for dependency injection.
"""

from typing import Any

import xgboost as xgb

from explainability.shap_explainer import create_shap_explainer
from explainability.dice_explainer import create_dice_explainer


def get_model() -> Any:
    """
    Return the loaded trained model instance for inference.

    Used as a FastAPI dependency. Loads the model on first call
    and caches it for subsequent requests.

    Returns:
        Trained model instance (XGBoost or fairness-constrained variant).
    """
    raise NotImplementedError("To be implemented")


def get_shap_explainer() -> Any:
    """
    Return the SHAP TreeExplainer initialized with the active model.

    Used as a FastAPI dependency for explanation endpoints.

    Returns:
        Configured shap.TreeExplainer instance.
    """
    raise NotImplementedError("To be implemented")


def get_dice_explainer() -> Any:
    """
    Return the DiCE explainer initialized with the active model and training data.

    Used as a FastAPI dependency for counterfactual generation.

    Returns:
        Configured dice_ml.Dice instance.
    """
    raise NotImplementedError("To be implemented")
