import pytest

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


def test_ssn_anonymization(anonymizer: PatternAnonymizer) -> None:
    text = "Patient SSN is 123-45-6789."
    scrubbed = anonymizer.scrub(text)
    assert "123-45-6789" not in scrubbed
    assert "[SSN]" in scrubbed


def test_phone_anonymization(anonymizer: PatternAnonymizer) -> None:
    text = "Contact 555-123-4567 or (800) 555-0199 or 9876543210 for follow up."
    scrubbed = anonymizer.scrub(text)
    assert "555-123-4567" not in scrubbed
    assert "9876543210" not in scrubbed
    assert scrubbed.count("[PHONE]") >= 3


def test_email_anonymization(anonymizer: PatternAnonymizer) -> None:
    text = "Email doctor.smith@hospital.org or patient.care@clinic.com."
    scrubbed = anonymizer.scrub(text)
    assert "doctor.smith@hospital.org" not in scrubbed
    assert "patient.care@clinic.com" not in scrubbed
    assert "[EMAIL]" in scrubbed


def test_mrn_anonymization(anonymizer: PatternAnonymizer) -> None:
    text = "Patient record MRN-994820 and MRN12345678 loaded."
    scrubbed = anonymizer.scrub(text)
    assert "994820" not in scrubbed
    assert "[MRN]" in scrubbed


def test_ner_masker_person_and_date(ner_masker: NERMasker) -> None:
    text = "John Smith was admitted on 2026-05-12."
    masked = ner_masker.mask_text(text)
    assert "John Smith" not in masked
    assert "[PATIENT_NAME]" in masked or "[DATE]" in masked


def test_privacy_pipeline_full(pipeline: PrivacyPipeline) -> None:
    raw = (
        "Patient John Smith (MRN-994820, SSN 123-45-6789) visited "
        "St. Jude Hospital in New York on 2026-05-12. Contact john.smith@email.com or 555-019-2831."
    )
    masked = pipeline.process(raw)

    # Ensure sensitive items are scrubbed
    assert "John Smith" not in masked
    assert "994820" not in masked
    assert "123-45-6789" not in masked
    assert "john.smith@email.com" not in masked
    assert "555-019-2831" not in masked

    # Ensure tokens are present
    assert any(
        token in masked
        for token in ["[PATIENT_NAME]", "[MRN]", "[SSN]", "[EMAIL]", "[PHONE]"]
    )
