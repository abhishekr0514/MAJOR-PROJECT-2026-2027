"""Integration tests for Member 1 backend API endpoints and FL orchestration."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.app import app
from app.core.database import SessionLocal
from app.core.security import hash_password
from app.features.hospitals.models import Hospital
from app.features.users.models import Role, User


@pytest.mark.asyncio
async def test_auth_and_prediction_endpoints():
    async with SessionLocal() as db:
        # Create test hospital
        hospital = Hospital(name="Test Hospital", license_code="HOSP_TEST")
        db.add(hospital)
        await db.commit()
        await db.refresh(hospital)

        # Create test super admin
        user = User(
            email="admin_test@medshield.org",
            hashed_password=hash_password("TestPassword123!"),
            full_name="Admin Test",
            role=Role.SUPER_ADMIN,
            hospital_id=hospital.id,
            is_active=True,
        )
        db.add(user)
        await db.commit()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as ac:
        # 1. Test POST /auth/login
        login_res = await ac.post(
            "/auth/login",
            data={
                "username": "admin_test@medshield.org",
                "password": "TestPassword123!",
            },
        )
        assert login_res.status_code == 200
        token_data = login_res.json()
        assert "access_token" in token_data
        token = token_data["access_token"]

        headers = {"Authorization": f"Bearer {token}"}

        # 2. Test GET /auth/me
        me_res = await ac.get("/auth/me", headers=headers)
        assert me_res.status_code == 200
        assert me_res.json()["email"] == "admin_test@medshield.org"

        # 3. Test POST /prediction/predict
        pred_res = await ac.post(
            "/prediction/predict",
            headers=headers,
            json={
                "patient_code": "PAT-TEST-01",
                "age": 62,
                "gender": "M",
                "blood_pressure_sys": 150,
                "blood_pressure_dia": 95,
                "cholesterol_mg_dl": 250.0,
                "clinical_text_masked": "Patient reports chest pain upon exertion.",
            },
        )
        assert pred_res.status_code == 201
        assert pred_res.json()["diagnosis"] == "High Risk"

        # 4. Test GET /federation/status
        fed_res = await ac.get("/federation/status", headers=headers)
        assert fed_res.status_code == 200
        assert "current_round" in fed_res.json()
