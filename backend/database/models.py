"""
Database ORM Models
SQLAlchemy declarative models for the EquiScore database.
"""

from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class DecisionLog(Base):
    """
    SQLAlchemy model for logging credit decisions.

    Stores each scoring decision with applicant features, model output,
    SHAP explanation, and fairness metadata for audit trail purposes.
    """

    __tablename__ = "decision_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    application_id = Column(String(50), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    applicant_features = Column(JSON, nullable=False)
    credit_score = Column(Integer, nullable=False)
    default_probability = Column(Float, nullable=False)
    risk_level = Column(String(10), nullable=False)
    shap_explanation = Column(JSON, nullable=True)
    model_version = Column(String(50), nullable=True)
    fairness_flags = Column(JSON, nullable=True)
