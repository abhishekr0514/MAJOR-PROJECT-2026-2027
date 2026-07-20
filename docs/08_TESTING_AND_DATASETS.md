# MedShield FL — Datasets, Mock Data Generator & Testing Blueprint (`Phase 8`)

This document details dataset integrations, synthetic multi-hospital dataset generators, automated testing suites, and final validation workflows (`/client/data/` & `/server/tests/`).

---

## 📊 Datasets & Data Modalities

The system uses three primary datasets for multimodal training:

| Data Modality | Source Dataset | Format | Extracted Features |
| :--- | :--- | :--- | :--- |
| **Tabular Lifestyle Data** | [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease) | CSV | Age, Sex, CP, Trestbps, Chol, Fbs, Restecg, Thalach, Exang, Oldpeak |
| **ECG Time-Series Signals** | [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/) | NumPy `.npy` | 12-lead time-series waveforms (500 Hz sampling rate, 10-sec windows) |
| **Clinical Text Notes** | Kaggle / MIMIC-III Clinical Notes | Plain Text | Anonymized medical history, clinical symptoms, doctor impressions |

---

## 🎲 Synthetic Multi-Hospital Data Generator (`client/data/generate_mock_data.py`)

Generates synthetic data splits simulating 3 distinct hospital client nodes (`Hospital Alpha`, `Hospital Beta`, `Hospital Gamma`).

```python
import numpy as np
import pandas as pd
import json
from pathlib import Path

def generate_hospital_datasets(output_dir: str = "client/data") -> None:
    """Generate synthetic tabular, ECG, and clinical text datasets for FL simulation."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    hospitals = ["hospital_alpha", "hospital_beta", "hospital_gamma"]
    num_samples_per_hospital = [150, 200, 180]

    sample_names = ["John Doe", "Alice Smith", "Robert Johnson", "Emily Davis"]
    sample_cities = ["New York", "Boston", "Chicago", "Seattle"]

    for idx, h_id in enumerate(hospitals):
        n_samples = num_samples_per_hospital[idx]
        
        # 1. Generate Synthetic Tabular Metrics
        df = pd.DataFrame({
            "patient_code": [f"PAT-{h_id[:3].upper()}-{i:04d}" for i in range(n_samples)],
            "age": np.random.randint(35, 80, size=n_samples),
            "gender": np.random.choice(["M", "F"], size=n_samples),
            "blood_pressure_sys": np.random.randint(110, 180, size=n_samples),
            "blood_pressure_dia": np.random.randint(70, 110, size=n_samples),
            "cholesterol_mg_dl": np.random.uniform(150, 320, size=n_samples).round(1),
            "fasting_bs_mg_dl": np.random.uniform(80, 160, size=n_samples).round(1),
            "diagnosis": np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]),
        })

        # 2. Add Synthetic Raw Text Notes with PII
        raw_texts = []
        for i in range(n_samples):
            name = np.random.choice(sample_names)
            city = np.random.choice(sample_cities)
            text = f"Patient {name} (MRN-{100000+i}) admitted in {city}. Reports tightness in chest and shortness of breath."
            raw_texts.append(text)
        df["raw_clinical_text"] = raw_texts

        # 3. Save CSV Dataset
        df.to_csv(out_path / f"{h_id}_data.csv", index=False)
        
        # 4. Generate Synthetic ECG Waveforms (n_samples, 12 leads, 1000 time-steps)
        ecg_signals = np.random.randn(n_samples, 12, 1000).astype(np.float32)
        np.save(out_path / f"{h_id}_ecg.npy", ecg_signals)

        print(f"✅ Generated dataset for {h_id}: {n_samples} samples.")

if __name__ == "__main__":
    generate_hospital_datasets()
```

---

## 🧪 Automated Testing Suite (`pytest`)

### 1. Privacy NER Masking Test (`client/tests/test_privacy_ner.py`)
```python
from client.privacy.pipeline import PrivacyPipeline

def test_ner_masking_zero_pii_leak():
    pipeline = PrivacyPipeline()
    sample_text = "Patient Alice Walker (SSN: 123-45-6789) visited Mercy Hospital on 2026-04-10."
    masked = pipeline.process(sample_text)

    assert "Alice Walker" not in masked
    assert "123-45-6789" not in masked
    assert "Mercy Hospital" not in masked
    assert "[PATIENT_NAME]" in masked or "[SSN]" in masked
```

### 2. PyTorch Model Forward Pass Test (`client/tests/test_ml_models.py`)
```python
import torch
from client.ml_models.full_model import MedShieldDiagnosticNet

def test_full_model_forward_pass():
    model = MedShieldDiagnosticNet()
    ecg = torch.randn(2, 12, 1000)
    input_ids = torch.randint(0, 1000, (2, 32))
    attn_mask = torch.ones(2, 32)
    tab = torch.randn(2, 10)

    logits = model(ecg, input_ids, attn_mask, tab)
    assert logits.shape == (2, 2)  # Binary classification logits
```

---

## 🚀 Final End-to-End System Verification Protocol

Run the following commands sequentially to verify the complete system:

```bash
# 1. Format and lint all Python files
make fix

# 2. Run automated test suite
make test

# 3. Generate synthetic hospital datasets
python client/data/generate_mock_data.py

# 4. Run database migrations & seed super admin
make migrate
make seed

# 5. Start FastAPI Backend Server
make run
```

---

## 🏁 Final Project Benchmark Targets

| Metric | Target Goal | Verification Method |
| :--- | :---: | :--- |
| **Privacy Anonymization Rate** | `100% PII Removal` | Regex & NER test suite |
| **Global FL Diagnosis Accuracy** | `> 88%` | Flower server evaluation across 5 rounds |
| **API Prediction Latency** | `< 250 ms` | FastAPI benchmark endpoint |
| **Counterfactual Convergence** | `< 3 iterations` | `DiCE` counterfactual generator test |
