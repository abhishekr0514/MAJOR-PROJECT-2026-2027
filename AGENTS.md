# MedShield FL — Sub-Agent Domains & Task Boundaries

To ensure clean modular implementation without cross-domain regression, tasks are divided into **4 specialized execution domains**:

---

## Domain 1: Backend API & Database Agent (`/server`)
* **Primary Scope**: Database schemas, Alembic migrations, REST API routers, authentication, and hospital node management.
* **Key Files**:
  * `server/app/core/` (`database.py`, `security.py`, `models.py`)
  * `server/app/features/users/`
  * `server/app/features/hospitals/`
  * `server/app/features/prediction/`
* **Rules**: Always update `server/app/core/models.py` when adding new tables so Alembic detects migrations.

---

## Domain 2: Privacy & Multimodal ML Agent (`/client`)
* **Primary Scope**: PII extraction and anonymization via NER, multimodal feature extractors, and fusion head architectures.
* **Key Files**:
  * `client/privacy/` (`ner_masker.py`, `anonymizer.py`)
  * `client/ml_models/` (`lstm_model.py`, `text_model.py`, `tabular_model.py`, `gnn_fusion.py`)
* **Rules**: Must ensure zero patient identity leak (PII) before features are passed into training loops or models.

---

## Domain 3: Federated Learning Agent (`/client` & `/server/app/features/federation`)
* **Primary Scope**: Flower (`flwr`) client node implementation, server strategy configuration (`FedAvg`/`FedProx`), and secure model update synchronization.
* **Key Files**:
  * `client/fl_client.py`
  * `server/app/features/federation/` (`fl_server.py`, `strategy.py`, `weights_store.py`)
* **Rules**: Ensure model parameters match PyTorch state dict representations across server and hospital nodes.

---

## Domain 4: Explainability, Causal AI & UI Agent (`/client/xai` & `/frontend`)
* **Primary Scope**: `DiCE` counterfactual generator, `DoWhy` causal graphs, and interactive React.js UI dashboard.
* **Key Files**:
  * `client/explainability/` (`counterfactual.py`, `causal_graph.py`)
  * `frontend/` (React components, diagnostic dashboard, clinician and patient portal views)
* **Rules**: Provide clear actionable recommendations (e.g. feature attribute impact and "what-if" parameter targets).
