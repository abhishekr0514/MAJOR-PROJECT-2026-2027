from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class PredictionCreateSchema(BaseModel):
    patient_code: str = Field(..., example="PAT-1042")
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., example="M")
    blood_pressure_sys: int = Field(..., ge=50, le=250)
    blood_pressure_dia: int = Field(..., ge=30, le=150)
    cholesterol_mg_dl: float = Field(..., ge=50.0, le=600.0)
    fasting_bs_mg_dl: float | None = Field(default=100.0, ge=10.0, le=500.0)
    clinical_text_masked: str
    ecg_signal_file_path: str | None = None
    chest_pain_type: int | None = None
    max_heart_rate: int | None = None
    exercise_angina: int | None = None


class PredictionResponseSchema(BaseModel):
    id: UUID
    patient_code: str
    risk_score: float
    diagnosis: str
    counterfactual_recommendations: list[dict] | None = None
    created_at: datetime

    class Config:
        from_attributes = True
