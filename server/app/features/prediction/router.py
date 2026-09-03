"""FastAPI Router for Prediction & Diagnostics."""

import uuid

from app.core.database import get_db
from app.features.patients.repository import PatientRepository
from app.features.prediction.repository import PredictionRepository
from app.features.prediction.schema import (
    PredictionCreateSchema,
    PredictionResponseSchema,
)
from app.features.prediction.service import process_prediction
from app.features.users.dependencies import get_current_active_user
from app.features.users.models import User
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

prediction_router = APIRouter()


@prediction_router.post(
    "/predict",
    response_model=PredictionResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest multimodal data and generate diagnostic prediction + XAI",
)
async def create_prediction(
    body: PredictionCreateSchema,
    user: User = Depends(get_current_active_user),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    return await process_prediction(db, body, user)


@prediction_router.get(
    "/history/{patient_id}",
    response_model=list[PredictionResponseSchema],
    summary="Get prediction history for a patient",
)
async def get_patient_history(
    patient_id: str,
    _: User = Depends(get_current_active_user),  # noqa: B008
    db: AsyncSession = Depends(get_db),  # noqa: B008
):
    repo = PredictionRepository(db)
    # Check if patient_id is a valid UUID, otherwise query by patient_code
    try:
        val_uuid = uuid.UUID(patient_id)
        history = await repo.get_history_by_patient(val_uuid)
    except ValueError:
        patient_repo = PatientRepository(db)
        patient = await patient_repo.get_one_by(patient_code=patient_id)
        if not patient:
            return []
        history = await repo.get_history_by_patient(patient.id)

    return [
        PredictionResponseSchema(
            id=item.id,
            patient_code=patient_id,
            risk_score=item.risk_score,
            diagnosis=item.diagnosis,
            counterfactual_recommendations=item.xai_counterfactuals,
            causal_impact=item.causal_impact,
            model_version=item.model_version,
            created_at=item.created_at,
        )
        for item in history
    ]
