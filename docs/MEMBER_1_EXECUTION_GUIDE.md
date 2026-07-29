# Member 1 Execution Guide — Systems, Database & Distributed FL Lead

This guide provides step-by-step instructions, blueprint references, and **AI Agent Pair-Programming Prompts** for **Member 1** to execute and complete their workload for **MedShield FL**.

---

## 📌 Domain Overview & Blueprint References

* **Role**: Systems Architecture, Database & Distributed FL Lead
* **Assigned Git Branch**: `feature/systems-db-fl`
* **Core Blueprints**:
  * [`docs/01_DATABASE_SCHEMA_AND_MODELS.md`](01_DATABASE_SCHEMA_AND_MODELS.md) — Database Schema & Models
  * [`docs/06_SERVER_API_SPEC.md`](06_SERVER_API_SPEC.md) — Server REST API & OpenAPI Specifications
  * [`docs/04_FEDERATED_LEARNING_FLWR.md`](04_FEDERATED_LEARNING_FLWR.md) — Flower FL Server & Client Orchestration

---

## 🛠️ Step-by-Step AI Agent Execution Workflow

### Step 1: Database Schemas & Alembic Migrations (`Phase 1`)

1. **Target Files**:
   * `server/app/core/models.py`
   * `server/migrations/versions/`
2. **AI Agent Prompt**:
   > *"You are acting as the Backend & Database AI Agent. Please read `docs/01_DATABASE_SCHEMA_AND_MODELS.md`. Implement the SQLAlchemy 2.0 ORM models (`Patient`, `ClinicalRecord`, `ECGRecord`, `Prediction`, `FLRound`, `FLModelUpdate`) inside `server/app/core/models.py`. Ensure all relationships, foreign keys, UUID primary keys, and indexes match the specification."*
3. **Execution Commands**:
   ```bash
   cd server
   # Generate migration script
   uv run alembic revision --autogenerate -m "Add core database models"
   # Apply migration to database
   uv run alembic upgrade head
   ```

---

### Step 2: OAuth2 JWT Authentication & RBAC (`Phase 6`)

1. **Target Files**:
   * `server/app/core/security.py`
   * `server/app/features/users/router.py`
2. **AI Agent Prompt**:
   > *"Please read `docs/06_SERVER_API_SPEC.md`. Implement OAuth2 password hashing (using bcrypt/passlib) and JWT token creation/verification in `server/app/core/security.py`. Implement the `/auth/login` and `/auth/me` endpoints in `server/app/features/users/router.py` supporting roles `SUPER_ADMIN`, `HOSPITAL_ADMIN`, and `CLINICIAN`."*
3. **Execution Commands**:
   ```bash
   uv run ruff check . --fix
   uv run ruff format .
   ```

---

### Step 3: FastAPI Endpoint Routers (`Phase 6`)

1. **Target Files**:
   * `server/app/features/prediction/router.py`
   * `server/app/features/prediction/schema.py`
   * `server/app/features/federation/router.py`
2. **AI Agent Prompt**:
   > *"Read `docs/06_SERVER_API_SPEC.md`. Implement Pydantic v2 validation schemas in `schema.py` for `/prediction/predict` and `/federation/status`. Implement FastAPI router endpoints for ingest prediction calls and FL round status monitoring."*

---

### Step 4: Distributed FL Server & Client Orchestration (`Phase 4`)

1. **Target Files**:
   * `server/app/features/federation/fl_server.py`
   * `client/fl_client.py`
2. **AI Agent Prompt**:
   > *"Read `docs/04_FEDERATED_LEARNING_FLWR.md`. Implement the central Flower FL server aggregation script (`fl_server.py`) using `FedAvg` and `FedProx` weight strategies. Implement the client-side `MedShieldFLClient` in `client/fl_client.py` using `flwr.client.NumPyClient` to handle local hospital training loops and weight serialization."*

---

## 🧪 Verification & Quality Check Commands

Run these commands before pushing your code to verify clean execution:

```bash
# 1. Format and lint all Python files
uv run ruff check .
uv run ruff format .

# 2. Run backend test suite
uv run pytest server/tests/

# 3. Test running the FastAPI dev server
uv run uvicorn server.main:app --reload --port 8000
```
