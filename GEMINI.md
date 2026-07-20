# MedShield FL — Project Guidelines & AI System Instructions

## Project Overview
**MedShield FL** is a privacy-preserving, explainable, multimodal federated learning framework for heart disease diagnosis. It enables hospital institutions to collaboratively train diagnostic models without exposing sensitive raw patient data, clinical text, or ECG readings.

- **Group ID**: 50
- **Domain**: Federated Learning, Healthcare AI, Privacy-Preserving ML, Multimodal Fusion, Explainable AI (XAI).

---

## Core Technology Stack

### 1. Backend Core (`/server`)
- **Framework**: FastAPI (Asynchronous API architecture)
- **Database ORM**: SQLAlchemy (Async Engine) with Alembic for schema migrations
- **Authentication**: JWT (JSON Web Tokens) + Password hashing with `bcrypt` / `passlib`
- **Role-Based Access Control (RBAC)**: `SUPER_ADMIN`, `HOSPITAL_ADMIN`, `CLINICIAN`

### 2. Client & Machine Learning (`/client`)
- **Federated Learning Framework**: Flower Framework (`flwr`) for client-server orchestration and `FedAvg`/`FedProx` weight aggregation
- **Privacy & Masking (NER)**: `spaCy` / BERT-based Named Entity Recognition for PII removal
- **ECG Signal Processing**: BiLSTM / Time-Series Transformer (PyTorch)
- **Clinical Text Processing**: BERT / Transformer Embeddings (HuggingFace Transformers)
- **Lifestyle / Tabular Data**: TabTransformer / Scikit-learn / XGBoost
- **Multimodal Fusion**: Graph Neural Networks (GNN) / Concatenation Fusion Layer
- **Explainability & Causal AI**: `DiCE` (Counterfactual Explanations) & `DoWhy` / `EconML` (Causal Inference)

### 3. Frontend & Dashboard (`/frontend` or `/client/ui`)
- **Framework**: React.js
- **Styling**: Modern, responsive CSS with glassmorphism, dynamic health gauges, and interactive counterfactual sliders.

---

## Strict Architectural Rules

1. **Privacy-First (Local Processing)**: Raw patient records, ECG signals, and clinical text notes MUST NEVER leave client hospital nodes unmasked. Named Entity Recognition (NER) masking occurs locally before any feature extraction or FL round.
2. **Federated Learning Principle**: The central server (`server/`) only collects and aggregates model weights/gradients. Raw training data remains strictly decentralized.
3. **Database & Migrations**: All database model changes in `server/app/features/` MUST be registered in `server/app/core/models.py` and applied via Alembic migrations (`uv run alembic upgrade head`). Never modify DB schemas manually.
4. **Code Quality, Linting & Type Hints**: All Python code MUST include strict type hints (`typing` module, `Pydantic` v2, Python 3.10+ syntax like `str | None` and `list[dict]`). All code MUST pass `uv run ruff check .` and `uv run ruff format .` clean without errors.
5. **No Placeholders in Final Features**: Avoid dummy static responses in backend services once a feature module is marked complete.

---

## Repository Structure

```
MAJOR-PROJECT-2026-2027/
├── docs/                        # Modular Technical Specifications & Execution Blueprints
├── report/                      # Project Research Report PDF & Presentation PPTX
├── server/                      # FastAPI Backend Server & Database Orchestration
│   ├── app/
│   │   ├── core/                # DB connection, Security, Base Models
│   │   └── features/            # Modular Domains (users, hospitals, federation, prediction)
│   ├── migrations/              # Alembic Database Migration Scripts
│   ├── main.py                  # Entrypoint for uvicorn server
│   └── seed.py                  # Database Seeding script
├── client/                      # Federated Client Node & ML Pipeline
│   ├── ml_models/               # BiLSTM, BERT, Tabular, & GNN models
│   ├── privacy/                 # spaCy / BERT NER masking pipeline
│   ├── fl_client.py             # Flower FL client node implementation
│   └── main.py                  # Client entrypoint script
├── GEMINI.md                    # Master AI System Instructions (This file)
├── SKILLS.md                    # Project Commands & Developer Procedures
└── AGENTS.md                    # Sub-agent Task & Domain Boundaries
```
