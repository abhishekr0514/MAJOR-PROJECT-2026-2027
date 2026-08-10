"""FastAPI Router for Prediction & Diagnostics."""

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.features.prediction.repository import PredictionRepository
from app.features.prediction.schema import (
    PredictionCreateSchema,
    PredictionResponseSchema,
)
from app.features.prediction.service import process_prediction
from app.features.users.dependencies import get_current_active_user
from app.features.users.models import User

prediction_router = APIRouter()


@prediction_router.post(
    "/predict",
    response_model=PredictionResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest multimodal data and generate diagnostic prediction + XAI",
)
async def create_prediction(
    body: PredictionCreateSchema,
    user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    return await process_prediction(db, body, user)


@prediction_router.get(
    "/history/{patient_id}",
    response_model=list[PredictionResponseSchema],
    summary="Get prediction history for a patient",
)
async def get_patient_history(
    patient_id: uuid.UUID,
    _: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    repo = PredictionRepository(db)
    history = await repo.get_history_by_patient(patient_id)
    return [
        PredictionResponseSchema(
            id=item.id,
            patient_code=str(item.patient_id),
            risk_score=item.risk_score,
            diagnosis=item.diagnosis,
            counterfactual_recommendations=item.xai_counterfactuals,
            causal_impact=item.causal_impact,
            model_version=item.model_version,
            created_at=item.created_at,
        )
        for item in history
    ]
