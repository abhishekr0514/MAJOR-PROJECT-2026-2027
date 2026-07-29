# Member 2 Execution Guide — Machine Learning & Explainable AI (XAI) Lead

This guide provides step-by-step instructions, blueprint references, and **AI Agent Pair-Programming Prompts** for **Member 2** to execute and complete their workload for **MedShield FL**.

---

## 📌 Domain Overview & Blueprint References

* **Role**: Pure Machine Learning & Explainable AI (XAI) Lead
* **Assigned Git Branch**: `feature/ml-models-xai`
* **Core Blueprints**:
  * [`docs/03_ML_MULTIMODAL_PIPELINE.md`](03_ML_MULTIMODAL_PIPELINE.md) — PyTorch Multimodal ML Models & Fusion
  * [`docs/05_EXPLAINABILITY_AND_CAUSAL_AI.md`](05_EXPLAINABILITY_AND_CAUSAL_AI.md) — DiCE & DoWhy Explainable Causal AI

---

## 🛠️ Step-by-Step AI Agent Execution Workflow

### Step 1: ECG BiLSTM & Tabular Models (`Phase 3`)

1. **Target Files**:
   * `client/ml_models/lstm_model.py`
   * `client/ml_models/tabular_model.py`
2. **AI Agent Prompt**:
   > *"You are acting as the Machine Learning AI Agent. Read `docs/03_ML_MULTIMODAL_PIPELINE.md`. Implement the 1D-Conv + Bidirectional LSTM PyTorch model in `lstm_model.py` for 12-lead ECG time-series signals. Implement the tabular feature encoder in `tabular_model.py` to process patient numerical and categorical lifestyle metrics."*

---

### Step 2: Clinical Text BERT Transformer Extractor (`Phase 3`)

1. **Target Files**:
   * `client/ml_models/text_model.py`
2. **AI Agent Prompt**:
   > *"Read `docs/03_ML_MULTIMODAL_PIPELINE.md`. Implement the `BioClinicalBERTFeatureExtractor` in `client/ml_models/text_model.py` using HuggingFace `transformers`. Extract fixed 768-dimensional embeddings from anonymized clinical text notes."*

---

### Step 3: Multimodal Neural Fusion Head (`Phase 3`)

1. **Target Files**:
   * `client/ml_models/gnn_fusion.py`
2. **AI Agent Prompt**:
   > *"Read `docs/03_ML_MULTIMODAL_PIPELINE.md`. Implement the GNN / Concatenation Fusion Neural Network in `gnn_fusion.py`. It should take ECG representations, text embeddings, and tabular feature vectors, pass them through a fusion layer, and output a binary diagnostic risk probability score for heart disease."*

---

### Step 4: DiCE Counterfactual Explainer (`Phase 5`)

1. **Target Files**:
   * `client/explainability/counterfactual.py`
2. **AI Agent Prompt**:
   > *"Read `docs/05_EXPLAINABILITY_AND_CAUSAL_AI.md`. Implement the `CounterfactualExplainer` class using the `DiCE` framework in `counterfactual.py`. Given a high-risk patient prediction, calculate actionable feature targets (e.g. reducing systolic BP or cholesterol) to lower predicted heart disease risk."*

---

### Step 5: DoWhy Causal AI Engine (`Phase 5`)

1. **Target Files**:
   * `client/explainability/causal_graph.py`
2. **AI Agent Prompt**:
   > *"Read `docs/05_EXPLAINABILITY_AND_CAUSAL_AI.md`. Implement the `CausalInferenceEngine` in `causal_graph.py` using `DoWhy`. Construct a Directed Acyclic Graph (DAG) representing clinical cause-effect assumptions and calculate average treatment effects (ATE) for target risk factors."*

---

## 🧪 Verification & Quality Check Commands

Run these commands before pushing your code to verify clean execution:

```bash
# 1. Format and lint all Python files
uv run ruff check .
uv run ruff format .

# 2. Test model forward pass & XAI pipeline
uv run pytest client/tests/test_ml_models.py
uv run pytest client/tests/test_xai.py
```
