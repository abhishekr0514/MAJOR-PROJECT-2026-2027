"""Decoupled standalone demonstration of the clinical text privacy and embeddings pipeline.

Performs simulated OCR -> Regex + spaCy NER PII masking -> BioClinicalBERT tokenizer ->
BioClinicalBERT model encoding -> 128-d output embedding projection.
"""

import sys

import torch

from client.ml_models.text_model import BioClinicalBERTFeatureExtractor
from client.privacy.pipeline import PrivacyPipeline


def main() -> None:
    print("====================================================")
    print("MEDSHIELD CLINICAL TEXT PRIVACY & ML DEMONSTRATION")
    print("====================================================")

    # 1. OCR Step
    print("\n[STEP 1] OCR Document Text Extraction")
    raw_ocr_text = (
        "Report Date: 2026-08-30. Clinic: Bangalore Medical Center. Patient: John Doe. "
        "Contact: +91-9876543210 or john.doe@email.com. SSN: 123-45-6789. MRN-7654321. "
        "Clinician Note: 56-year-old male presenting with chest pain, palpitations, and dyspnea. "
        "ECG shows potential sinus tachycardia with ST elevation. High risk of myocardial infarction."
    )
    print(f"Loaded Raw Note (OCR Extracted):\n{raw_ocr_text}")

    # 2. Anonymization / PII scrubbing
    print("\n[STEP 2] PII Scrubbing (Regex + NER Masking)")
    pipeline = PrivacyPipeline()
    scrubbed_text = pipeline.process(raw_ocr_text)
    print(f"Scrubbed Note Output:\n{scrubbed_text}")

    # Verify PII was masked
    print("\nVerifying masked placeholders:")
    placeholders = ["[PATIENT_NAME]", "[PHONE]", "[EMAIL]", "[SSN]", "[MRN]"]
    for placeholder in placeholders:
        if placeholder in scrubbed_text or (
            placeholder == "[PATIENT_NAME]" and "[PATIENT_NAME]" in scrubbed_text
        ):
            print(f"  - [VERIFIED] Masked placeholder {placeholder} detected.")
        else:
            # Note: PERSON -> [PATIENT_NAME] or [PERSON]
            if placeholder == "[PATIENT_NAME]" and "[PERSON]" in scrubbed_text:
                print("  - [VERIFIED] Masked placeholder [PERSON] detected.")
            else:
                print(
                    f"  - [WARNING] Expected placeholder {placeholder} was not found (might not be matched)."
                )

    # 3. Tokenizer & BERT Loading
    print("\n[STEP 3] BioClinicalBERT Tokenization")
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = BioClinicalBERTFeatureExtractor.get_tokenizer(model_name)
    if tokenizer is None:
        print(
            "ERROR: HuggingFace tokenizer could not be loaded or instantiated. Aborting."
        )
        sys.exit(1)

    tokens = tokenizer(
        [scrubbed_text],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]
    print(f"Tokenized Input IDs shape: {tuple(input_ids.shape)}")
    print(f"Tokenized Attention Mask shape: {tuple(attention_mask.shape)}")

    # 4. Feature Extraction & Projection
    print("\n[STEP 4] BioClinicalBERT Feature Encoding & Projection")
    # Set lazy_load=False to load and verify weights immediately
    extractor = BioClinicalBERTFeatureExtractor(
        pretrained_model=model_name,
        output_dim=128,
        lazy_load=False,
    )

    if extractor.bert is None:
        print("ERROR: Feature Extractor model weights failed to load. Aborting.")
        sys.exit(1)

    extractor.eval()
    with torch.no_grad():
        embeddings = extractor(input_ids, attention_mask, verbose=True)

    print(
        f"\nFinal projected text embedding vector (128-d) shape: {tuple(embeddings.shape)}"
    )
    print("Text pipeline demonstration completed successfully!")


if __name__ == "__main__":
    main()
