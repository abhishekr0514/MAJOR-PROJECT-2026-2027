import uuid

from app.core.database import get_db
from app.features.hospitals.models import Hospital
from app.features.patients.models import Patient
from app.features.prediction.models import Prediction
from app.features.prediction.schema import (
    PredictionCreateSchema,
    PredictionResponseSchema,
)
from app.features.users.dependencies import get_current_active_user
from app.features.users.models import Role, User
from app.features.users.permissions import RoleChecker
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

prediction_router = APIRouter()

# Restrict to CLINICIAN or SUPER_ADMIN for testing
_require_clinician_or_admin = RoleChecker([Role.CLINICIAN, Role.SUPER_ADMIN])


def calculate_risk(
    age: int, bp_sys: int, bp_dia: int, cholesterol: float, fasting_bs: float | None
) -> float:
    base = 0.15
    age_factor = max(0.0, (age - 35) * 0.005)
    sys_factor = max(0.0, (bp_sys - 110) * 0.004)
    dia_factor = max(0.0, (bp_dia - 70) * 0.003)
    chol_factor = max(0.0, (cholesterol - 150) * 0.0015)
    fbs_factor = 0.08 if (fasting_bs and fasting_bs > 120.0) else 0.0

    raw_risk = base + age_factor + sys_factor + dia_factor + chol_factor + fbs_factor
    return min(0.95, max(0.05, raw_risk))


def get_diagnosis(risk: float) -> str:
    if risk >= 0.70:
        return "High Risk"
    elif risk >= 0.40:
        return "Moderate Risk"
    else:
        return "Low Risk"


@prediction_router.post(
    "/predict",
    response_model=PredictionResponseSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Compute prediction risk score and generate counterfactual recommendations",
)
async def predict(
    body: PredictionCreateSchema,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    # Determine hospital_id to associate with patient
    hospital_id = current_user.hospital_id
    if not hospital_id:
        # Fallback: find the first available hospital in DB
        result_hosp = await db.execute(select(Hospital))
        first_hosp = result_hosp.scalars().first()
        if not first_hosp:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No hospital registered. Create a hospital page/record first.",
            )
        hospital_id = first_hosp.id

    # Lookup or create patient
    stmt = select(Patient).where(Patient.patient_code == body.patient_code)
    db_result = await db.execute(stmt)
    patient = db_result.scalars().first()

    if not patient:
        patient = Patient(
            id=uuid.uuid4(),
            patient_code=body.patient_code,
            age=body.age,
            gender=body.gender,
            hospital_id=hospital_id,
        )
        db.add(patient)
        await db.commit()
        await db.refresh(patient)

    # Compute risk score and diagnosis
    risk_score = calculate_risk(
        age=body.age,
        bp_sys=body.blood_pressure_sys,
        bp_dia=body.blood_pressure_dia,
        cholesterol=body.cholesterol_mg_dl,
        fasting_bs=body.fasting_bs_mg_dl,
    )
    diagnosis = get_diagnosis(risk_score)

    # Generate counterfactuals
    # Option 1: Reduce bp_sys and cholesterol
    target_bp = max(110, body.blood_pressure_sys - 20)
    target_chol = max(150.0, body.cholesterol_mg_dl - 50.0)
    opt1_risk = calculate_risk(
        body.age, target_bp, body.blood_pressure_dia, target_chol, body.fasting_bs_mg_dl
    )

    # Option 2: Reduce bp_sys and bp_dia
    target_bp_dia = max(70, body.blood_pressure_dia - 15)
    opt2_risk = calculate_risk(
        body.age,
        target_bp,
        target_bp_dia,
        body.cholesterol_mg_dl,
        body.fasting_bs_mg_dl,
    )

    counterfactuals = [
        {
            "option": 1,
            "target_changes": {
                "cholesterol_mg_dl": round(target_chol, 1),
                "blood_pressure_sys": target_bp,
            },
            "predicted_new_risk": round(opt1_risk, 3),
            "predicted_new_diagnosis": get_diagnosis(opt1_risk),
        },
        {
            "option": 2,
            "target_changes": {
                "blood_pressure_sys": target_bp,
                "blood_pressure_dia": target_bp_dia,
            },
            "predicted_new_risk": round(opt2_risk, 3),
            "predicted_new_diagnosis": get_diagnosis(opt2_risk),
        },
    ]

    prediction = Prediction(
        id=uuid.uuid4(),
        patient_id=patient.id,
        risk_score=round(risk_score, 3),
        diagnosis=diagnosis,
        xai_counterfactuals={"recommendations": counterfactuals},
        model_version="1.0.0",
    )
    db.add(prediction)
    await db.commit()
    await db.refresh(prediction)

    return PredictionResponseSchema(
        id=prediction.id,
        patient_code=body.patient_code,
        risk_score=prediction.risk_score,
        diagnosis=prediction.diagnosis,
        counterfactual_recommendations=counterfactuals,
        created_at=prediction.created_at,
    )


@prediction_router.get(
    "/history/{patient_code}",
    response_model=list[PredictionResponseSchema],
    summary="Get prediction records history for a patient",
)
async def get_history(
    patient_code: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Patient).where(Patient.patient_code == patient_code)
    db_res = await db.execute(stmt)
    patient = db_res.scalars().first()
    if not patient:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found"
        )

    pred_stmt = (
        select(Prediction)
        .where(Prediction.patient_id == patient.id)
        .order_by(Prediction.created_at.desc())
    )
    pred_res = await db.execute(pred_stmt)
    predictions = pred_res.scalars().all()

    response = []
    for pred in predictions:
        cfs = (
            pred.xai_counterfactuals.get("recommendations")
            if pred.xai_counterfactuals
            else []
        )
        response.append(
            PredictionResponseSchema(
                id=pred.id,
                patient_code=patient_code,
                risk_score=pred.risk_score,
                diagnosis=pred.diagnosis,
                counterfactual_recommendations=cfs,
                created_at=pred.created_at,
            )
        )
    return response
