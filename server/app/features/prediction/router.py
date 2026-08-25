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
    "/history/{patient_identifier}",
    response_model=list[PredictionResponseSchema],
    summary="Get prediction history for a patient",
)
async def get_patient_history(
    patient_identifier: str,
    _: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    from app.features.patients.repository import PatientRepository

    patient_repo = PatientRepository(db)
    pred_repo = PredictionRepository(db)

    patient_obj = None
    try:
        pid = uuid.UUID(patient_identifier)
        patient_obj = await patient_repo.get_by_id(pid)
    except ValueError:
        patient_obj = await patient_repo.get_one_by(patient_code=patient_identifier)

    if not patient_obj:
        return []

    history = await pred_repo.get_history_by_patient(patient_obj.id)
    return [
        PredictionResponseSchema(
            id=item.id,
            patient_code=patient_obj.patient_code,
            risk_score=item.risk_score,
            diagnosis=item.diagnosis,
            counterfactual_recommendations=item.xai_counterfactuals,
            causal_impact=item.causal_impact,
            model_version=item.model_version,
            created_at=item.created_at,
        )
        for item in history
    ]


from pydantic import BaseModel


class MaskTextRequest(BaseModel):
    text: str


class MaskTextResponse(BaseModel):
    raw_text: str
    masked_text: str
    engine: str


@prediction_router.post(
    "/mask-text",
    response_model=MaskTextResponse,
    summary="Scrub PII from clinical text notes using Member 2's spaCy NER model",
)
async def mask_text_endpoint(body: MaskTextRequest):
    import sys
    from pathlib import Path

    root_dir = Path(__file__).resolve().parents[4]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    try:
        from client.privacy.ner_masker import NERMasker

        masker = NERMasker()
        masked = masker.mask_text(body.text)
        engine = "spaCy Transformer (en_core_web_sm)"
    except Exception as e:
        import re

        masked = re.sub(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", "[PATIENT_NAME]", body.text)
        engine = f"Fallback ({e})"

    return MaskTextResponse(raw_text=body.text, masked_text=masked, engine=engine)

