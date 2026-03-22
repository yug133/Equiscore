"""
Prediction Route
POST /predict — Score an applicant and return credit decision with SHAP explanation.
"""

from fastapi import APIRouter, Depends

from api.schemas import PredictRequest, PredictResponse
from api.dependencies import get_model, get_shap_explainer

router = APIRouter()


@router.post("/", response_model=PredictResponse)
async def predict_credit_score(
    request: PredictRequest,
    model=Depends(get_model),
    explainer=Depends(get_shap_explainer),
) -> PredictResponse:
    """
    Score a loan applicant and return credit decision.

    Takes the 18-field applicant form, runs the fairness-constrained
    XGBoost model, computes SHAP explanation, scales the probability
    to a 300-900 credit score, and returns the full response.

    Args:
        request: PredictRequest with applicant features.
        model: Injected trained model instance.
        explainer: Injected SHAP explainer instance.

    Returns:
        PredictResponse with credit score, risk level, and SHAP explanation.
    """
    raise NotImplementedError("To be implemented")
