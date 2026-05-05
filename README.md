# Federated Heart Diagnosis System

A production-grade, privacy-preserving medical AI platform for heart disease diagnosis. This project uses **Federated Learning** to train machine learning models across decentralized medical institutions without moving sensitive patient data.

---

## 🏗️ Architecture

- **Backend**: FastAPI (Python 3.12+)
- **Database**: PostgreSQL with Async SQLAlchemy 2.0
- **Migrations**: Alembic (Professional version control for your schema)
- **Patterns**: Repository Pattern + Service Layer (Clean Architecture)
- **Auth**: JWT-based Authentication with Role-Based Access Control (RBAC)

---

## 👥 Role Hierarchy

| Role | Access Level | Responsibilities |
| :--- | :--- | :--- |
| **Super Admin** | Global | Manage Hospitals, Global Model Aggregation, System Audit |
| **Hospital Admin** | Local (Node) | Manage Clinicians within their hospital, Local Training Rounds |
| **Clinician** | End-User | Patient Diagnosis, Viewing Predictions, Local Data Management |

---

## 🛠️ Setup & Development

This project uses `uv` for lightning-fast dependency management and a `Makefile` for streamlined commands.

### 1. Prerequisites
- [uv](https://github.com/astral-sh/uv) installed
- PostgreSQL running locally

### 2. Configuration
Copy the example environment file and fill in your details:
```bash
cp server/.example.env server/.env
```

### 3. Installation
```bash
make install
```

### 4. Database Setup
```bash
# Apply migrations
make migrate

# Create the first Super Admin (configured in .env)
make seed
```

### 5. Running the App
```bash
make run
```
The API will be available at `http://localhost:8000`.  
View the interactive docs at `http://localhost:8000/docs`.

---

## 📂 Core Commands Reference

| Command | Description |
| :--- | :--- |
| `make run` | Start the development server |
| `make migrate` | Apply all pending migrations |
| `make makemigration m="msg"` | Generate a new database migration |
| `make seed` | Initialize the system with a Super Admin |
| `make lint` / `make format` | Keep the code clean (Ruff) |

---

## 🚀 Future Roadmap
- [ ] **Federation Engine**: Implementing client-server orchestration for model training.
- [ ] **Prediction Service**: Real-time heart disease inference using trained models.
- [ ] **Email Invitations**: Secure token-based user onboarding.
- [ ] **Frontend**: Interactive dashboard for clinicians and admins.
