"""
Pydantic Request/Response Schemas
Typed models for API request validation and response serialization.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for POST /predict endpoint."""

    age: int = Field(..., description="Applicant age in years")
    gender: str = Field(..., description="Applicant gender (M/F)")
    income: float = Field(..., description="Annual income in INR")
    employment_type: str = Field(..., description="Employment type (Salaried/Self-employed/etc)")
    occupation_type: str = Field(..., description="Occupation category")
    education_type: str = Field(..., description="Education level")
    family_status: str = Field(..., description="Marital / family status")
    housing_type: str = Field(..., description="Housing situation type")
    region_rating: int = Field(..., description="Region rating (1-3)")
    own_car: bool = Field(..., description="Whether applicant owns a car")
    own_realty: bool = Field(..., description="Whether applicant owns property")
    children_count: int = Field(..., description="Number of children")
    family_members: int = Field(..., description="Number of family members")
    credit_amount: float = Field(..., description="Requested credit amount")
    annuity_amount: float = Field(..., description="Loan annuity amount")
    goods_price: float = Field(..., description="Price of goods for which loan is requested")
    ext_source_1: Optional[float] = Field(None, description="External source score 1")
    ext_source_2: Optional[float] = Field(None, description="External source score 2")


class PredictResponse(BaseModel):
    """Response schema for POST /predict endpoint."""

    application_id: str = Field(..., description="Unique application identifier")
    credit_score: int = Field(..., description="Credit score on 300-900 scale")
    default_probability: float = Field(..., description="Probability of default (0-1)")
    risk_level: str = Field(..., description="Risk category: LOW / MEDIUM / HIGH")
    shap_explanation: Dict[str, float] = Field(..., description="SHAP values per feature")
    top_factors: List[str] = Field(..., description="Top 5 contributing features")


class AuditResponse(BaseModel):
    """Response schema for GET /audit endpoint."""

    model_name: str = Field(..., description="Name of the audited model")
    overall_metrics: Dict[str, float] = Field(..., description="AUC-ROC, Gini, KS metrics")
    fairness_metrics: Dict[str, Dict[str, float]] = Field(
        ..., description="DPG, EOD, DIR per subgroup"
    )
    fairness_flags: List[str] = Field(..., description="Subgroups violating fairness thresholds")


class ImproveRequest(BaseModel):
    """Request schema for POST /improve endpoint."""

    application_id: str = Field(..., description="Application ID to generate tips for")
    num_tips: int = Field(3, description="Number of counterfactual tips to generate")


class ImproveResponse(BaseModel):
    """Response schema for POST /improve endpoint."""

    application_id: str = Field(..., description="Application identifier")
    current_score: int = Field(..., description="Current credit score")
    tips: List[Dict[str, Any]] = Field(
        ..., description="List of actionable improvement suggestions from DiCE"
    )
    potential_score: int = Field(
        ..., description="Estimated score after following suggestions"
    )
