"""
EquiScore - Fair and Explainable Credit Scoring System
FastAPI Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import predict, audit, improve
from database.connection import get_engine, create_tables
from utils.logger import setup_logger

app = FastAPI(
    title="EquiScore API",
    description="Fair and Explainable Credit Scoring for Thin-File Applicants",
    version="0.1.0",
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(audit.router, prefix="/audit", tags=["Audit"])
app.include_router(improve.router, prefix="/improve", tags=["Improvement"])


@app.on_event("startup")
async def startup_event() -> None:
    """
    Initialize database tables and logger on application startup.
    """
    raise NotImplementedError("To be implemented")


@app.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint to verify the API is running.

    Returns:
        dict with status key indicating service health.
    """
    return {"status": "ok"}
