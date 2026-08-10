import argparse

import spacy
import spacy.cli


class NERMasker:
    """Named Entity Recognition (NER) engine for masking patient PII in clinical text notes."""

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        """Initialize spaCy NLP model for entity detection."""
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)

        self.replacement_map: dict[str, str] = {
            "PERSON": "[PATIENT_NAME]",
            "DATE": "[DATE]",
            "TIME": "[DATE]",
            "GPE": "[LOCATION]",
            "LOC": "[LOCATION]",
            "ORG": "[HOSPITAL_NAME]",
        }

        self.medical_whitelist: set[str] = {
            "angina", "ischemic", "ischemia", "infarction", "dyspnea", "arrhythmia",
            "hypertension", "hypotension", "tachycardia", "bradycardia", "stenosis",
            "cardiomyopathy", "atherosclerosis", "edema", "syncope", "troponin", "st-segment"
        }

    def mask_text(self, text: str) -> str:
        """Scrub named entities and structured PII from raw clinical text.

        Entities are processed in reverse order of their start character position
        to prevent index offsets from shifting during replacement.
        """
        import re

        # 1. spaCy NER on raw text (with medical terms whitelisted)
        doc = self.nlp(text)
        masked_text = text

        entities = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)
        for ent in entities:
            if ent.text.lower() in self.medical_whitelist:
                continue
            if ent.label_ in self.replacement_map:
                placeholder = self.replacement_map[ent.label_]
                masked_text = (
                    masked_text[: ent.start_char]
                    + placeholder
                    + masked_text[ent.end_char :]
                )

        # 2. Structured PII Scrubbing (Email, Phone, SSN, MRN)
        masked_text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            "[EMAIL]",
            masked_text,
        )
        masked_text = re.sub(
            r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b",
            "[SSN_ID]",
            masked_text,
        )
        masked_text = re.sub(
            r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}|\b\d{7,15}\b",
            "[PHONE_NUMBER]",
            masked_text,
        )

        return masked_text


def main() -> None:
    """CLI entrypoint for testing NER text masking directly."""
    parser = argparse.ArgumentParser(
        description="Scrub named entities (PII) from clinical text notes."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Raw clinical text input to mask.",
    )
    args = parser.parse_args()

    masker = NERMasker()
    result = masker.mask_text(args.input)
    print("Masked Output:")
    print(result)


if __name__ == "__main__":
    main()
