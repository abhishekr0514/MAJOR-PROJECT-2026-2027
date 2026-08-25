import os
import sys

# Add project root and server subdirectories to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
server_dir = os.path.join(root_dir, "server")

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if server_dir not in sys.path:
    sys.path.insert(0, server_dir)

import asyncio

# Set mock environment variables for settings validation before any app imports
os.environ["POSTGRES_USER"] = "postgres"
os.environ["POSTGRES_PASSWORD"] = "password"
os.environ["POSTGRES_SERVER"] = "localhost"
os.environ["POSTGRES_PORT"] = "5432"
os.environ["POSTGRES_DB"] = "medshield"
os.environ["AUTH_SECRET"] = "superSecretKey123ForTesting"
os.environ["FIRST_SUPER_ADMIN_EMAIL"] = "admin@medshield.org"
os.environ["FIRST_SUPER_ADMIN_PASSWORD"] = "securepass123"

import pytest
from app.app import app
from app.core.database import Base, get_db
from app.features.hospitals.models import Hospital
from app.features.users.models import Role, User
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# In-memory SQLite engine
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestingSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db():
    async with engine.begin() as conn:
        # Create all tables on-the-fly
        await conn.run_sync(Base.metadata.create_all)


# Initialize database schemas synchronously before any tests start
asyncio.run(init_db())


async def override_get_db():
    async with TestingSessionLocal() as session:
        yield session


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def setup_test_entities():
    """Initializes a hospital and admin/clinician users for tests."""

    async def _setup():
        async with TestingSessionLocal() as session:
            # 1. Create a Test Hospital in database
            hospital = Hospital(
                name="Mercy General Hospital",
                license_code="HOSP-MERCY-0099",
                address="100 Medical Way, Boston, MA",
            )
            session.add(hospital)
            await session.commit()
            await session.refresh(hospital)

            # 2. Create Clinician and Super Admin users
            # Note: password is saved as plain text hashed using bcrypt in actual service,
            # here we hash passwords manually to match auth_router's authentication behavior.
            from app.core.security import hash_password

            hashed_pass = hash_password("securepass123")

            clinician = User(
                email="clinician@hospital.org",
                full_name="Dr. Sarah Connor",
                hashed_password=hashed_pass,
                role=Role.CLINICIAN,
                hospital_id=hospital.id,
                is_active=True,
            )
            admin = User(
                email="admin@medshield.org",
                full_name="MedShield Coordinator",
                hashed_password=hashed_pass,
                role=Role.SUPER_ADMIN,
                hospital_id=None,
                is_active=True,
            )
            session.add(clinician)
            session.add(admin)
            await session.commit()

            return {
                "hospital_id": str(hospital.id),
                "clinician_email": clinician.email,
                "admin_email": admin.email,
            }

    return asyncio.run(_setup())


def test_user_authentication_flow(client: TestClient, setup_test_entities: dict):
    # 1. Test Login to get Access Token
    response = client.post(
        "/auth/login",
        data={
            "username": setup_test_entities["clinician_email"],
            "password": "securepass123",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

    token = data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Test profile endpoint /auth/me
    me_resp = client.get("/auth/me", headers=headers)
    assert me_resp.status_code == 200
    me_data = me_resp.json()
    assert me_data["email"] == setup_test_entities["clinician_email"]
    assert me_data["role"] == "clinician"
    assert me_data["hospital_id"] == setup_test_entities["hospital_id"]


def test_signup_auth_validation(client: TestClient):
    signup_payload = {
        "email": "new.clinician@hospital.org",
        "full_name": "Dr. John Watson",
        "password": "mypassword101",
        "role": "clinician",
    }
    response = client.post("/auth/signup", json=signup_payload)
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "new.clinician@hospital.org"
    assert "id" in data


def test_diagnostic_prediction_routing(client: TestClient, setup_test_entities: dict):
    # Login as clinician
    login_resp = client.post(
        "/auth/login",
        data={
            "username": setup_test_entities["clinician_email"],
            "password": "securepass123",
        },
    )
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Predict Request Payload
    payload = {
        "patient_code": "PAT-BOSTON-8012",
        "age": 60,
        "gender": "M",
        "blood_pressure_sys": 150,
        "blood_pressure_dia": 95,
        "cholesterol_mg_dl": 265.0,
        "fasting_bs_mg_dl": 130.0,
        "clinical_text_masked": "Patient [PATIENT_NAME] (MRN-[MRN]) reports exerts angina.",
        "ecg_signal_file_path": None,
    }

    # Call predict REST API Contract
    pred_resp = client.post("/prediction/predict", json=payload, headers=headers)
    assert pred_resp.status_code == 201
    pred_data = pred_resp.json()

    # Contract validation checks
    assert "id" in pred_data
    assert pred_data["patient_code"] == "PAT-BOSTON-8012"
    assert pred_data["risk_score"] > 0
    assert pred_data["diagnosis"] in ["Low Risk", "Moderate Risk", "High Risk"]
    assert "counterfactual_recommendations" in pred_data
    assert len(pred_data["counterfactual_recommendations"]) >= 1

    # Verify predictions save correctly in database and history works
    hist_resp = client.get("/prediction/history/PAT-BOSTON-8012", headers=headers)
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert len(hist_data) >= 1
    assert hist_data[0]["patient_code"] == "PAT-BOSTON-8012"


def test_federation_training_rounds(client: TestClient, setup_test_entities: dict):
    # 1. Login as Clinician (fails to trigger FL rounds - Permission check)
    clinician_login = client.post(
        "/auth/login",
        data={
            "username": setup_test_entities["clinician_email"],
            "password": "securepass123",
        },
    )
    clinician_token = clinician_login.json()["access_token"]
    clinician_headers = {"Authorization": f"Bearer {clinician_token}"}

    fl_payload = {"num_rounds": 3, "min_clients": 2}
    forbidden_resp = client.post(
        "/federation/rounds/start", json=fl_payload, headers=clinician_headers
    )
    assert forbidden_resp.status_code == 403  # Forbidden for clinician

    # 2. Login as Super Admin (succeeds to trigger rounds)
    admin_login = client.post(
        "/auth/login",
        data={
            "username": setup_test_entities["admin_email"],
            "password": "securepass123",
        },
    )
    admin_token = admin_login.json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    start_resp = client.post(
        "/federation/rounds/start", json=fl_payload, headers=admin_headers
    )
    assert start_resp.status_code == 202
    start_data = start_resp.json()
    assert "round_id" in start_data
    assert "triggered successfully" in start_data["message"]

    # 3. Fetch status and evaluate updates
    status_resp = client.get("/federation/status", headers=admin_headers)
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["current_round"] >= 1
    assert status_data["status"] == "IN_PROGRESS"
    assert status_data["global_accuracy"] > 0.8
