"""Integration tests for Member 1 backend API endpoints and FL orchestration."""

import asyncio
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.app import app
from app.core.database import Base, get_db

# In-memory SQLite engine for clean test isolation with StaticPool
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    poolclass=StaticPool,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)


async def init_test_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


asyncio.run(init_test_db())


async def override_get_db():
    async with TestingSessionLocal() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_auth_and_prediction_endpoints(client: TestClient):
    # 1. Signup test user
    signup_res = client.post(
        "/auth/signup",
        json={
            "email": "admin_test@medshield.org",
            "full_name": "Admin Test",
            "password": "TestPassword123!",
            "role": "super_admin",
        },
    )
    assert signup_res.status_code == 201

    # 2. Test POST /auth/login
    login_res = client.post(
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

    # 3. Test GET /auth/me
    me_res = client.get("/auth/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "admin_test@medshield.org"

    # 4. Test POST /prediction/predict
    pred_res = client.post(
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
    assert pred_res.json()["diagnosis"] in ["Low Risk", "Moderate Risk", "High Risk"]

    # 5. Test GET /federation/status
    fed_res = client.get("/federation/status", headers=headers)
    assert fed_res.status_code == 200
    assert "current_round" in fed_res.json()
