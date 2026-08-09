import os
import sys

# Add project root directory to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from client.data.generate_mock_data import generate_hospital_datasets
from client.privacy.anonymizer import PatternAnonymizer
from client.privacy.ner_masker import NERMasker
from client.privacy.pipeline import PrivacyPipeline


@pytest.fixture
def anonymizer() -> PatternAnonymizer:
    return PatternAnonymizer()


@pytest.fixture
def ner_masker() -> NERMasker:
    return NERMasker()


@pytest.fixture
def pipeline() -> PrivacyPipeline:
    return PrivacyPipeline()


@pytest.fixture(scope="module")
def mock_dataset_dir():
    # Setup temporary directory for synthetic datasets
    temp_dir = tempfile.mkdtemp()
    generate_hospital_datasets(output_dir=temp_dir, seed=42)
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


def test_ssn_regex_scrubbing(anonymizer: PatternAnonymizer):
    raw = "The patient's SSN is 999-12-8872."
    scrubbed = anonymizer.scrub(raw)
    assert "999-12-8872" not in scrubbed
    assert "[SSN]" in scrubbed


def test_phone_regex_scrubbing(anonymizer: PatternAnonymizer):
    raw = "Call doctor at 555-555-1234 or clinic line: (800) 123-4567."
    scrubbed = anonymizer.scrub(raw)
    assert "555-555-1234" not in scrubbed
    assert "123-4567" not in scrubbed
    assert "[PHONE]" in scrubbed
    assert scrubbed.count("[PHONE]") == 2


def test_email_regex_scrubbing(anonymizer: PatternAnonymizer):
    raw = "Send clinical files to test.patient_john-doe@sub-hospital.org."
    scrubbed = anonymizer.scrub(raw)
    assert "test.patient_john-doe@sub-hospital.org" not in scrubbed
    assert "[EMAIL]" in scrubbed


def test_mrn_regex_scrubbing(anonymizer: PatternAnonymizer):
    raw = "Patient MRN-9988221 was diagnosed. Alternative code is MRN1029302."
    scrubbed = anonymizer.scrub(raw)
    assert "9988221" not in scrubbed
    assert "1029302" not in scrubbed
    assert "[MRN]" in scrubbed


def test_ner_masker_names_and_locations(ner_masker: NERMasker):
    raw = "Patient Emily Davis visited City Health Medical Center in Chicago."
    scrubbed = ner_masker.mask_text(raw)
    # Ensure raw names/locations are masked
    assert "Emily Davis" not in scrubbed
    assert "Chicago" not in scrubbed
    assert any(
        placeholder in scrubbed
        for placeholder in ["[PATIENT_NAME]", "[LOCATION]", "[HOSPITAL_NAME]"]
    )


def test_complete_privacy_pipeline(pipeline: PrivacyPipeline):
    raw = (
        "Patient Alice Smith (SSN: 111-22-3333, MRN-4455667) admitted in Boston on 2026-08-04. "
        "Reach via alice.smith@example.org or phone 555-987-6543. Reports chest pain."
    )
    scrubbed = pipeline.process(raw)

    # Negative assertions ensuring zero PII leak
    assert "Alice Smith" not in scrubbed
    assert "111-22-3333" not in scrubbed
    assert "4455667" not in scrubbed
    assert "Boston" not in scrubbed
    assert "alice.smith@example.org" not in scrubbed
    assert "555-987-6543" not in scrubbed

    # Positive assertions ensuring placeholders are substituted
    assert any(
        token in scrubbed
        for token in [
            "[PATIENT_NAME]",
            "[SSN]",
            "[MRN]",
            "[LOCATION]",
            "[EMAIL]",
            "[PHONE]",
            "[DATE]",
        ]
    )


def test_dataset_file_structure(mock_dataset_dir: Path):
    hospitals = ["hospital_a", "hospital_b", "hospital_c"]
    for h in hospitals:
        csv_file = mock_dataset_dir / f"{h}_data.csv"
        npy_file = mock_dataset_dir / f"{h}_ecg.npy"
        assert csv_file.exists()
        assert npy_file.exists()
        assert csv_file.stat().st_size > 0
        assert npy_file.stat().st_size > 0


def test_dataset_tabular_metrics_validity(mock_dataset_dir: Path):

    df = pd.read_csv(mock_dataset_dir / "hospital_a_data.csv")

    # Column completeness
    expected_cols = {
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
    assert expected_cols.issubset(set(df.columns))

    # Metric distributions and validity constraints
    assert df["age"].min() >= 35
    assert df["age"].max() <= 80
    assert set(df["gender"].unique()).issubset({"M", "F"})
    assert df["blood_pressure_sys"].min() >= 110
    assert df["blood_pressure_sys"].max() <= 180
    assert df["blood_pressure_dia"].min() >= 70
    assert df["blood_pressure_dia"].max() <= 110
    assert df["cholesterol_mg_dl"].min() >= 150.0
    assert df["cholesterol_mg_dl"].max() <= 320.0
    assert set(df["diagnosis"].unique()).issubset({0, 1})
    assert set(df["chest_pain_type"].unique()).issubset({0, 1, 2, 3})
    assert set(df["exercise_angina"].unique()).issubset({0, 1})


def test_dataset_ecg_signal_shapes(mock_dataset_dir: Path):
    ecg_a = np.load(mock_dataset_dir / "hospital_a_ecg.npy")
    ecg_b = np.load(mock_dataset_dir / "hospital_b_ecg.npy")
    ecg_c = np.load(mock_dataset_dir / "hospital_c_ecg.npy")

    # Shapes: (count, leads, samples)
    assert ecg_a.shape == (150, 12, 1000)
    assert ecg_b.shape == (200, 12, 1000)
    assert ecg_c.shape == (180, 12, 1000)
    assert ecg_a.dtype == np.float32
