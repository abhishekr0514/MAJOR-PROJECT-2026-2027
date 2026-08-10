# 🛡️ MedShield FL — Comprehensive Multi-Stakeholder User Guide

**MedShield FL** is a privacy-preserving, explainable, multimodal federated learning framework for heart disease diagnosis. It enables hospital institutions to collaboratively train diagnostic models without exposing sensitive raw patient data, clinical text, or ECG readings.

This guide provides end-to-end operational instructions for all project stakeholders.

---

## 👥 Stakeholder Role Matrix

| Stakeholder Role | Access Level | Primary Objectives & Tasks |
| :--- | :--- | :--- |
| **1. Super Administrator** | `SUPER_ADMIN` | System setup, DB migrations, hospital node key management, FL round orchestration, API SDK sync. |
| **2. Hospital Node Operator** | `HOSPITAL_ADMIN` | Deploying local Flower FL client nodes, local dataset verification, monitoring local node telemetry. |
| **3. Clinician / Cardiologist** | `CLINICIAN` | Submitting multimodal diagnostic records, reviewing XAI counterfactuals, using "What-If" risk sliders. |
| **4. Patient** | `PATIENT` | Reviewing personal anonymized health summaries, tracking cardiovascular stability trends, lifestyle targets. |

---

## 🚀 Quickstart & Environment Setup

### System Prerequisites
- **Operating System**: Windows 11 (WSL2 recommended or Native PowerShell), Linux (Ubuntu 22.04+ / Fedora), or macOS
- **Python Runtime**: Python 3.10+ (managed via `uv` or Python installer)
- **Node.js Environment**: Node.js v18+ and `npm`
- **Database Engine**: PostgreSQL v14+ (or Docker Desktop / PostgreSQL for Windows)

### First-Time Initialization Commands

```bash
# 1. Clone repository
git clone https://github.com/chetanuchiha16/medshield-fl.git
cd MAJOR-PROJECT-2026-2027

# 2. Setup environment variables
cp server/.env.example server/.env

# 3. Apply database migrations & seed initial Super Admin
cd server
uv run alembic upgrade head
uv run python seed.py
cd ..

# 4. Install frontend dependencies
cd frontend
npm install
cd ..
```

---

## 🛠️ Stakeholder Operational Guides

### 1. 👑 Super Administrator Guide

As a Super Admin, you oversee system health, hospital institution registration, and global FL model training rounds.

#### Key Operations:

##### A. Starting the Infrastructure

**Option 1: Linux / macOS / Windows WSL2 (Using `make`)**
```bash
# Terminal 1: Run FastAPI backend server (Port 8000)
make run

# Terminal 2: Run React frontend dashboard (Port 5173)
make frontend

# Terminal 3: Run Flower FL Central Server (Port 8080)
make fl-server
```

**Option 2: Windows 11 Native (PowerShell / Command Prompt)**
```powershell
# Terminal 1: Run FastAPI backend server
cd server
uv run uvicorn main:app --reload --port 8000

# Terminal 2: Run React frontend dashboard
cd frontend
npm run dev

# Terminal 3: Run Flower FL Central Server
cd server
uv run python -m app.features.federation.fl_server
```

##### B. Authenticating via the React UI
1. Navigate to `http://localhost:5173`.
2. Click **`Sign In (JWT)`** in the top-right navigation header.
3. Enter Super Admin credentials:
   - **Email**: `admin@heart.com`
   - **Password**: `change-me-immediately`
4. The header badge will update to **`Connected (admin@heart.com)`**.

##### C. Registering Hospital Nodes
1. Navigate to the **Admin Console** tab in the UI.
2. Select **`Hospital Nodes Management`**.
3. Click **`+ Register Hospital Node`**.
4. Enter hospital name and allocated records. A secure license key (e.g. `MSFL-A91X-8801`) will be issued.

##### D. Triggering a Federated Learning Training Round
1. In **Admin Console**, select your **Aggregation Strategy**:
   - `FedAvg` (Standard Federated Averaging)
   - `FedProx` (Proximal Regularization for Non-IID Hospital Datasets)
2. Set minimum required client nodes (e.g. `2`).
3. Click **`Trigger Training Round`**. The central server will coordinate weight updates across connected client nodes.

##### E. Regenerating the API SDK (On Backend Changes)
Whenever FastAPI routes or schemas update:
```bash
make generate-api
```

---

### 2. 🏥 Hospital Node Operator Guide

Hospital operators manage local institution client nodes. Raw patient data and ECG signals remain strictly inside your hospital's secure network.

#### Key Operations:

##### A. Running a Federated Client Node
To connect your hospital node to the central FL server:
```bash
# Linux / macOS / Windows WSL2:
make fl-client

# Windows 11 Native (PowerShell):
cd client
uv run python fl_client.py --server 127.0.0.1:8080 --hospital-id hospital_alpha
```

##### B. Local Privacy & NER Masking Verification
Before local model training or feature extraction begins, Named Entity Recognition (NER) strips sensitive patient identifiers:
- **Redacted Fields**: Patient Names, Social Security Numbers (SSN), Medical Record Numbers (MRN), Phone Numbers, Email Addresses, and exact dates.
- **Verification Script**:
  ```bash
  cd client
  uv run python privacy/ner_masker.py
  ```

---

### 3. 🩺 Clinician & Cardiologist Guide

Clinicians use MedShield FL to evaluate patient heart disease risk using multimodal features (ECG signals + Clinical Text Notes + Tabular Vitals) with Explainable AI (XAI).

#### Key Operations:

##### A. Submitting a Diagnostic Evaluation
1. Select the **Clinician Dashboard** tab in the React UI.
2. Fill in patient parameters:
   - **Patient ID**: e.g., `PAT-1042`
   - **Age & Gender**: e.g., `58 yrs`, `Male`
   - **Vitals**: Systolic/Diastolic Blood Pressure, Serum Cholesterol, Fasting Blood Sugar.
   - **Clinical Notes**: Enter clinical observations (e.g. *"Patient reports exertional chest tightness"*).
3. Observe the **Privacy NER Pipeline** preview — names and dates are automatically masked in real-time.
4. Click **`Secure Predict & Evaluate`**.

##### B. Interpreting Multimodal Diagnostic Results
- **Radial Risk Gauge**: Displays overall cardiovascular disease risk score ($0\% - 100\%$).
- **12-Lead ECG Waveform**: Interactive time-series visualizer showing ST-segment elevation/depression.
- **Model Signature**: Identifies fusion model version (e.g. `1.2.0-BiLSTM+BERT`).

##### C. Using the Counterfactual "What-If" Simulator
1. Scroll to the **Counterfactual Target Simulator** panel.
2. Adjust target Systolic BP or Cholesterol sliders.
3. Observe projected risk reductions in real-time to help formulate patient treatment plans (e.g., *"Lowering Systolic BP to 120 mmHg reduces risk from 64% to 22%"*).

---

### 4. 👤 Patient Guide

Patients access anonymized health metrics and personalized lifestyle recommendations.

#### Key Operations:

##### A. Viewing Cardiovascular Summary
1. Select the **Patient Portal** tab in the UI.
2. Review your **Cardiovascular Attributes Table** (Demographics, Blood Pressure, Cholesterol).

##### B. Tracking Health Stability Trends
- Examine the **Cardiovascular Stability Indicator Track** to view monthly risk trend lines.

##### C. Actionable Interventions
- Review personalized targets:
  - **Nutrition**: Fiber and Omega-3 guidelines to lower cholesterol.
  - **Cardiovascular**: DASH diet sodium limits for blood pressure.
  - **Metabolic**: Post-meal walking targets for blood sugar control.

---

## 🔧 Troubleshooting & Reference

| Issue / Error | Cause | Resolution |
| :--- | :--- | :--- |
| `POST /prediction/predict 401 Unauthorized` | JWT token not present in browser storage. | Click **`Sign In (JWT)`** in header and log in with `admin@heart.com`. |
| `POST /prediction/predict 404 Not Found` | Request hitting Vite dev server port (5173) instead of API port (8000). | Ensure `make run` is running and API base URL is set to `http://localhost:8000`. |
| `IntegrityError: duplicate key` in tests | Pre-existing test entries in DB. | Run `uv run alembic upgrade head` or re-seed test DB. |
| `Untyped function calls` in TS | IDE background type inference. | Clean build via `make generate-api` and `npm run build`. |

---

## 📜 Port Reference Table

| Service / Component | Protocol / Port | Entrypoint File |
| :--- | :--- | :--- |
| **FastAPI Backend Server** | `HTTP / 8000` | `server/main.py` |
| **React Frontend App** | `HTTP / 5173` | `frontend/src/main.jsx` |
| **Flower FL Server** | `gRPC / 8080` | `server/app/features/federation/fl_server.py` |
| **PostgreSQL Database** | `TCP / 5432` | `server/app/core/database.py` |

---

*© 2026-2027 MedShield FL Consortium. All rights and specifications reserved.*
