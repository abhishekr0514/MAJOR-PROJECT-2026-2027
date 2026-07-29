# Member 3 Execution Guide — Privacy Engine, Datasets & Frontend UI Lead

This guide provides step-by-step instructions, blueprint references, and **AI Agent Pair-Programming Prompts** for **Member 3** to execute and complete their workload for **MedShield FL**.

---

## 📌 Domain Overview & Blueprint References

* **Role**: Privacy Engine, Datasets & Frontend UI Lead
* **Assigned Git Branch**: `feature/privacy-datasets-reactui`
* **Core Blueprints**:
  * [`docs/02_PRIVACY_NER_MODULE.md`](02_PRIVACY_NER_MODULE.md) — spaCy / BERT Privacy NER Anonymizer
  * [`docs/07_FRONTEND_REACT_DASHBOARD.md`](07_FRONTEND_REACT_DASHBOARD.md) — Vite + React.js UI Dashboard
  * [`docs/08_TESTING_AND_DATASETS.md`](08_TESTING_AND_DATASETS.md) — Medical Datasets & Test Suite

---

## 🛠️ Step-by-Step AI Agent Execution Workflow

### Step 1: Privacy NER Masker & Anonymizer (`Phase 2`)

1. **Target Files**:
   * `client/privacy/ner_masker.py`
   * `client/privacy/anonymizer.py`
2. **AI Agent Prompt**:
   > *"You are acting as the Privacy & Security AI Agent. Read `docs/02_PRIVACY_NER_MODULE.md`. Implement `ner_masker.py` using `spaCy` (or BERT NER) to extract patient names, dates, and locations from clinical text notes. Implement regex fallbacks in `anonymizer.py` to replace SSNs, phone numbers, and emails with anonymized tokens like `[PATIENT_NAME]` and `[DATE]`."*

---

### Step 2: Datasets & Multi-Hospital Mock Data Generator (`Phase 8`)

1. **Target Files**:
   * `client/data/`
   * `generate_mock_data.py`
2. **AI Agent Prompt**:
   > *"Read `docs/08_TESTING_AND_DATASETS.md`. Build a multi-hospital synthetic patient dataset generator in `generate_mock_data.py`. Generate realistic patient tabular metrics, clinical text notes, and ECG 12-lead signal arrays partitioned into Hospital A, Hospital B, and Hospital C datasets."*

---

### Step 3: Vite + React.js Setup & Design System (`Phase 7`)

1. **Target Directory**: `frontend/`
2. **Execution & AI Prompt**:
   ```bash
   npx -y create-vite@latest frontend --template react
   cd frontend
   npm install axios recharts lucide-react clsx tailwindcss
   ```
   > *"Read `docs/07_FRONTEND_REACT_DASHBOARD.md`. Configure the Vite + React.js design system in `frontend/src/assets/`. Implement a modern dark-mode glassmorphism layout with dynamic emerald green to coral red health risk styling."*

---

### Step 4: React UI Components & Dashboards (`Phase 7`)

1. **Target Files**:
   * `frontend/src/components/clinician/CounterfactualSlider.jsx`
   * `frontend/src/components/admin/FLTrainingMonitor.jsx`
   * `frontend/src/pages/ClinicianDashboard.jsx`
   * `frontend/src/pages/PatientPortal.jsx`
   * `frontend/src/pages/AdminDashboard.jsx`
2. **AI Agent Prompt**:
   > *"Read `docs/07_FRONTEND_REACT_DASHBOARD.md`. Implement the interactive `CounterfactualSlider.jsx` and `FLTrainingMonitor.jsx` components using Recharts. Build the `ClinicianDashboard.jsx` view with diagnostic forms and ECG waveform visualizers, the `PatientPortal.jsx` view with health summaries, and the `AdminDashboard.jsx` view with FL node monitors."*

---

### Step 5: System Test Suite & API Verification (`Phase 8`)

1. **Target Files**:
   * `tests/test_privacy.py`
   * `tests/test_end_to_end.py`
2. **AI Agent Prompt**:
   > *"Read `docs/08_TESTING_AND_DATASETS.md`. Write `pytest` integration test suites in `tests/` verifying that zero patient PII leaks through the privacy pipeline, dataset formatting is valid, and backend REST API contracts match frontend Axios requests."*

---

## 🧪 Verification & Quality Check Commands

Run these commands before pushing your code to verify clean execution:

```bash
# 1. Format and lint Python code
uv run ruff check .
uv run ruff format .

# 2. Run Privacy and End-to-End Pytest suite
uv run pytest tests/

# 3. Test building and running React frontend
cd frontend
npm run dev
```
