# MedShield FL — Server API Specifications & OpenAPI Contracts (`Phase 6`)

This document specifies the FastAPI REST API route endpoints, request/response Pydantic v2 schemas, role-based authorization rules, and status codes (`server/app/features/`).

---

## 🌐 API Overview & Router Architecture

All routes require **JWT Bearer Token Authentication** (except `/auth/login`).

```
FastAPI Server (Port 8000)
├── /auth               --> Login & Session Management
├── /users              --> User Management (Super Admin & Hospital Admin)
├── /hospitals          --> Federated Hospital Node Management
├── /prediction         --> Multimodal Data Ingest, Prediction & XAI
└── /federation         --> Federated Learning Round Orchestration & Weights Sync
```

---

## 📡 Endpoint Contracts

### 1. Authentication Router (`/auth`)

#### `POST /auth/login`
Authenticates a user and returns a OAuth2 Bearer Access Token.
- **Request Body**: `OAuth2PasswordRequestForm` (`username: email`, `password: string`)
- **Response `200 OK`**:
  ```json
  {
    "access_token": "eyJhbGciOiJIUzI1Ni...",
    "token_type": "bearer"
  }
  ```
- **Errors**: `401 Unauthorized` (Invalid credentials)

#### `GET /auth/me`
Fetches current logged-in user profile & role.
- **Header**: `Authorization: Bearer <token>`
- **Response `200 OK`**:
  ```json
  {
    "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
    "email": "doctor@hospital.org",
    "full_name": "Dr. Sarah Connor",
    "role": "clinician",
    "hospital_id": "8b3f12a4-1100-4311-b112-4211940188bc"
  }
  ```

---

### 2. Diagnostic & Prediction Router (`/prediction`)

#### `POST /prediction/predict`
Ingests ECG signals, masked text, and tabular metrics to generate a diagnostic risk score & XAI counterfactuals.
- **Permission**: Clinicians (`Role.CLINICIAN`)
- **Multipart Form / Request Body**:
  ```json
  {
    "patient_code": "PAT-99412",
    "age": 58,
    "gender": "M",
    "blood_pressure_sys": 145,
    "blood_pressure_dia": 90,
    "cholesterol_mg_dl": 240,
    "fasting_bs_mg_dl": 110,
    "clinical_text_masked": "Patient reports severe angina upon exertion. [PATIENT_NAME] on [DATE].",
    "ecg_signal_file_id": "ecg_file_uuid_or_path"
  }
  ```
- **Response `201 Created`**:
  ```json
  {
    "prediction_id": "d1a89f64-5717-4562-b3fc-2c963f66afa6",
    "patient_code": "PAT-99412",
    "risk_score": 0.82,
    "diagnosis": "High Risk",
    "counterfactual_recommendations": [
      {
        "option": 1,
        "target_changes": {
          "cholesterol_mg_dl": 190,
          "blood_pressure_sys": 125
        },
        "predicted_new_risk": 0.25,
        "predicted_new_diagnosis": "Low Risk"
      }
    ],
    "created_at": "2026-07-20T18:30:00Z"
  }
  ```

#### `GET /prediction/history/{patient_id}`
Retrieves prediction history & risk trend for a specific patient.
- **Response `200 OK`**: `list[PredictionSchema]`

---

### 3. Federated Learning Router (`/federation`)

#### `GET /federation/status`
Checks current FL training round status, active hospital clients, and global model version.
- **Response `200 OK`**:
  ```json
  {
    "current_round": 4,
    "total_rounds": 10,
    "status": "IN_PROGRESS",
    "global_accuracy": 0.915,
    "participating_hospitals_count": 3
  }
  ```

#### `POST /federation/rounds/start`
Triggers a new FL training round across connected hospital nodes.
- **Permission**: Super Admin (`Role.SUPER_ADMIN`)
- **Request Body**:
  ```json
  {
    "num_rounds": 5,
    "min_clients": 2
  }
  ```
- **Response `202 Accepted`**:
  ```json
  {
    "message": "FL Round 5 triggered successfully.",
    "round_id": "f5a89f64-5717-4562-b3fc-2c963f66afa6"
  }
  ```

---

## 🔒 Pydantic Schema Definitions (`server/app/features/prediction/schema.py`)

```python
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime


class PredictionCreateSchema(BaseModel):
    patient_code: str = Field(..., example="PAT-1042")
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., example="M")
    blood_pressure_sys: int = Field(..., ge=50, le=250)
    blood_pressure_dia: int = Field(..., ge=30, le=150)
    cholesterol_mg_dl: float = Field(..., ge=50.0, le=600.0)
    clinical_text_masked: str
    ecg_signal_file_path: str | None = None


class PredictionResponseSchema(BaseModel):
    id: UUID
    patient_code: str
    risk_score: float
    diagnosis: str
    counterfactual_recommendations: list[dict] | None = None
    created_at: datetime

    class Config:
        from_attributes = True
```

---

## ✅ Phase 6 Verification Checklist
- [ ] Authentication endpoints (`/auth/login`, `/auth/me`) fully specified
- [ ] User & Hospital management CRUD endpoints defined
- [ ] Multimodal prediction endpoint (`/prediction/predict`) accepts tabular, text, and ECG inputs
- [ ] FL orchestration endpoints (`/federation/status`, `/federation/rounds/start`) specified
- [ ] Pydantic v2 schemas include strict validation ranges (`ge`, `le`, `Field`)
