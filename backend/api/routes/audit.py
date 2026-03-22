"""
Audit Route
GET /audit — Return fairness audit report with DPG, EOD, DIR metrics.
"""

from fastapi import APIRouter

from api.schemas import AuditResponse

router = APIRouter()


@router.get("/", response_model=AuditResponse)
async def get_audit_report() -> AuditResponse:
    """
    Return the latest fairness audit report.

    Computes or retrieves cached fairness metrics (DPG, EOD, DIR) for
    all sensitive subgroups including intersectional groups. Flags
    subgroups that violate fairness thresholds.

    Returns:
        AuditResponse with overall metrics, subgroup metrics, and flags.
    """
    raise NotImplementedError("To be implemented")
