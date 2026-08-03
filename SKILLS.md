# MedShield FL — Project Developer Skills & Operational Workflows

This document outlines the standard procedural workflows, CLI commands, and execution scripts for building, testing, and running **MedShield FL**.

---

## 1. Environment & Setup Commands

### Backend Server (`/server`)
```bash
# Navigate to server directory
cd server

# Install dependencies using uv
uv sync

# Run database migrations (Alembic)
uv run alembic upgrade head

# Generate a new migration revision after model changes
uv run alembic revision --autogenerate -m "describe changes"

# Seed default Super Admin & initial test hospitals
uv run python seed.py

# Start local FastAPI server with hot-reload (Port 8000)
uv run python main.py
```

### Client Node & ML Pipeline (`/client`)
```bash
# Navigate to client directory
cd client

# Install dependencies using uv / pip
uv sync

# Run NER Privacy Masker test on clinical notes
python -m privacy.ner_masker --input "Patient John Doe (ID: 1042) diagnosed with hypertension."

# Start a Flower FL Client node connecting to server
python fl_client.py --server 127.0.0.1:8080 --hospital-id hospital_01
```

### Makefile Quick Shortcuts (Root Directory)
```bash
make help          # View all available make targets
make run           # Start FastAPI backend server
make migrate       # Run DB migrations (alembic upgrade head)
make makemigration m="add_patients_table"  # Create new migration
make seed          # Seed default database users & hospitals
make lint          # Run ruff lint check across project
make format        # Auto-format project code with ruff
make fix           # Fix ruff lint errors automatically
make test          # Run pytest test suite
make clean         # Clean cache directories (__pycache__, .ruff_cache)
```

---

## 2. Testing Workflows

### Running Backend Unit & Integration Tests
```bash
cd server
uv run pytest tests/ -v
```

### Code Quality, Linting & Formatting (`ruff`)
```bash
# Check code for linting errors and auto-fix imports/formatting
uv run ruff check . --fix

# Auto-format code across server/ and client/
uv run ruff format .
```

### Simulating Federated Learning Rounds Locally
```bash
# Terminal 1: Launch FL Aggregation Server
python -m server.app.features.federation.fl_server --rounds 5 --port 8080

# Terminal 2 & 3: Launch 2 Simulated Hospital FL Clients
python client/fl_client.py --hospital-id hospital_alpha --data client/data/hospital_a.csv
python client/fl_client.py --hospital-id hospital_beta --data client/data/hospital_b.csv
```

---

## 3. Recommended Development Order

1. **Database Schema & Models**: Always update SQLAlchemy models in `server/app/features/` and run `uv run alembic upgrade head`.
2. **Backend API Endpoint**: Implement service + repository + router in `server/app/features/<feature>/`.
3. **Client ML / Privacy Unit**: Develop and unit-test standalone module in `client/privacy/` or `client/ml_models/`.
4. **Integration**: Link client FL / prediction calls to FastAPI endpoints.
