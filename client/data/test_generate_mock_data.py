import os
import sys
from pathlib import Path

# Add project root directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import numpy as np
import pandas as pd
import pytest

from client.data.generate_mock_data import generate_hospital_datasets
from client.privacy.pipeline import PrivacyPipeline


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    generate_hospital_datasets(output_dir=str(tmp_path), seed=123)
    return tmp_path


def test_generated_files_exist(data_dir: Path) -> None:
    expected_files = [
        "hospital_a_data.csv",
        "hospital_a_ecg.npy",
        "hospital_b_data.csv",
        "hospital_b_ecg.npy",
        "hospital_c_data.csv",
        "hospital_c_ecg.npy",
        "hospital_alpha_data.csv",
        "hospital_alpha_ecg.npy",
        "hospital_beta_data.csv",
        "hospital_beta_ecg.npy",
        "hospital_gamma_data.csv",
        "hospital_gamma_ecg.npy",
    ]
    for filename in expected_files:
        filepath = data_dir / filename
        assert filepath.exists(), f"Missing file: {filename}"
        assert filepath.stat().st_size > 0, f"Empty file: {filename}"


def test_tabular_data_schema(data_dir: Path) -> None:
    df_a = pd.read_csv(data_dir / "hospital_a_data.csv")
    expected_columns = {
        "patient_code",
        "age",
        "gender",
        "blood_pressure_sys",
        "blood_pressure_dia",
        "cholesterol_mg_dl",
        "fasting_bs_mg_dl",
        "chest_pain_type",
        "max_heart_rate",
        "exercise_angina",
        "diagnosis",
        "raw_clinical_text",
    }
    assert expected_columns.issubset(set(df_a.columns))
    assert len(df_a) == 150
    assert set(df_a["diagnosis"].unique()).issubset({0, 1})


def test_ecg_tensor_shapes(data_dir: Path) -> None:
    ecg_a = np.load(data_dir / "hospital_a_ecg.npy")
    ecg_b = np.load(data_dir / "hospital_b_ecg.npy")
    ecg_c = np.load(data_dir / "hospital_c_ecg.npy")

    assert ecg_a.shape == (150, 12, 1000)
    assert ecg_b.shape == (200, 12, 1000)
    assert ecg_c.shape == (180, 12, 1000)
    assert ecg_a.dtype == np.float32


def test_generated_text_privacy_pipeline_scrubbing(data_dir: Path) -> None:
    df_a = pd.read_csv(data_dir / "hospital_a_data.csv")
    pipeline = PrivacyPipeline()

    sample_raw_text = df_a["raw_clinical_text"].iloc[0]
    masked_text = pipeline.process(sample_raw_text)

    # Ensure unmasked text had PII tokens/patterns replaced
    assert sample_raw_text != masked_text
    assert "@example.com" not in masked_text
    assert "[EMAIL]" in masked_text or "[PATIENT_NAME]" in masked_text
