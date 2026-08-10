"""Pydantic v2 validation schemas for prediction feature."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PredictionCreateSchema(BaseModel):
    patient_code: str = Field(
        ..., description="Anonymized patient code (e.g. PAT-1042)"
    )
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., description="Gender indicator (e.g. M, F, Other)")
    blood_pressure_sys: int = Field(..., ge=50, le=250)
    blood_pressure_dia: int = Field(..., ge=30, le=150)
    cholesterol_mg_dl: float = Field(..., ge=50.0, le=600.0)
    fasting_bs_mg_dl: float | None = Field(default=None, ge=0.0, le=500.0)
    clinical_text_masked: str = Field(..., description="NER masked clinical notes")
    ecg_signal_file_path: str | None = Field(
        default=None, description="Local ECG signal file path"
    )


class CounterfactualOption(BaseModel):
    option: int
    target_changes: dict[str, float | int]
    predicted_new_risk: float
    predicted_new_diagnosis: str


class PredictionResponseSchema(BaseModel):
    id: uuid.UUID
    patient_code: str
    risk_score: float
    diagnosis: str
    counterfactual_recommendations: list[dict] | None = None
    causal_impact: dict | None = None
    model_version: str = "1.0.0"
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
