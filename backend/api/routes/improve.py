"""
Improve Route
POST /improve — Generate counterfactual improvement tips using DiCE.
"""

from fastapi import APIRouter, Depends

from api.schemas import ImproveRequest, ImproveResponse
from api.dependencies import get_dice_explainer

router = APIRouter()


@router.post("/", response_model=ImproveResponse)
async def get_improvement_tips(
    request: ImproveRequest,
    dice_explainer=Depends(get_dice_explainer),
) -> ImproveResponse:
    """
    Generate actionable improvement suggestions for a rejected or
    low-scoring applicant using DiCE counterfactual explanations.

    Retrieves the applicant's decision from the database, runs DiCE
    on actionable features only, and returns specific, realistic tips
    the applicant can follow to improve their credit score.

    Args:
        request: ImproveRequest with application_id and num_tips.
        dice_explainer: Injected DiCE explainer instance.

    Returns:
        ImproveResponse with current score, tips, and potential score.
    """
    raise NotImplementedError("To be implemented")
