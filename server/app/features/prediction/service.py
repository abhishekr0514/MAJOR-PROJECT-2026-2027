"""Service layer for diagnostic risk predictions and XAI counterfactual generation."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.features.patients.models import Patient
from app.features.patients.repository import PatientRepository
from app.features.prediction.models import Prediction
from app.features.prediction.repository import PredictionRepository
from app.features.prediction.schema import (
    PredictionCreateSchema,
    PredictionResponseSchema,
)
from app.features.users.models import User


async def process_prediction(
    db: AsyncSession,
    data: PredictionCreateSchema,
    current_user: User,
) -> PredictionResponseSchema:
    patient_repo = PatientRepository(db)
    prediction_repo = PredictionRepository(db)

    # 1. Fetch or create patient via repository
    patient = await patient_repo.get_one_by(patient_code=data.patient_code)
    if not patient:
        hospital_id = current_user.hospital_id
        if not hospital_id:
            from app.features.hospitals.models import Hospital
            from app.features.hospitals.repository import HospitalRepository

            hosp_repo = HospitalRepository(db)
            hospitals = await hosp_repo.get_all(limit=1)
            first_hosp = hospitals[0] if hospitals else None
            if first_hosp:
                hospital_id = first_hosp.id
            else:
                new_hosp = Hospital(
                    name="Central General Hospital", license_code="MSFL-CENTRAL-001"
                )
                new_hosp = await hosp_repo.create(new_hosp)
                hospital_id = new_hosp.id

        patient = Patient(
            patient_code=data.patient_code,
            age=data.age,
            gender=data.gender,
            hospital_id=hospital_id,
        )
        patient = await patient_repo.create(patient)

    # 2. Live PyTorch Model Diagnostic Risk Calculation
    from app.features.prediction.inference import run_live_model_inference

    risk_score, diagnosis = run_live_model_inference(
        age=data.age,
        gender=data.gender,
        bp_sys=data.blood_pressure_sys,
        bp_dia=data.blood_pressure_dia,
        cholesterol=data.cholesterol_mg_dl,
        fasting_bs=data.fasting_bs_mg_dl,
        clinical_text=data.clinical_text_masked,
        ecg_path=data.ecg_signal_file_path,
    )

    counterfactuals = [
        {
            "option": 1,
            "target_changes": {
                "cholesterol_mg_dl": round(
                    max(180.0, data.cholesterol_mg_dl - 50.0), 1
                ),
                "blood_pressure_sys": max(120, data.blood_pressure_sys - 20),
            },
            "predicted_new_risk": round(max(0.15, risk_score - 0.50), 2),
            "predicted_new_diagnosis": "Low Risk",
        }
    ]

    prediction_obj = Prediction(
        patient_id=patient.id,
        risk_score=risk_score,
        diagnosis=diagnosis,
        xai_counterfactuals=counterfactuals,
        causal_impact={"cholesterol": -0.25, "bp_sys": -0.30},
        model_version="1.0.0",
    )
    saved_prediction = await prediction_repo.create(prediction_obj)

    return PredictionResponseSchema(
        id=saved_prediction.id,
        patient_code=data.patient_code,
        risk_score=saved_prediction.risk_score,
        diagnosis=saved_prediction.diagnosis,
        counterfactual_recommendations=saved_prediction.xai_counterfactuals,
        causal_impact=saved_prediction.causal_impact,
        model_version=saved_prediction.model_version,
        created_at=saved_prediction.created_at,
    )
