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

    def mask_text(self, text: str) -> str:
        """Scrub named entities from raw clinical text.

        Entities are processed in reverse order of their start character position
        to prevent index offsets from shifting during replacement.
        """
        doc = self.nlp(text)
        masked_text = text

        # Sort entities in reverse position order to avoid offset shift
        entities = sorted(doc.ents, key=lambda e: e.start_char, reverse=True)
        for ent in entities:
            if ent.label_ in self.replacement_map:
                placeholder = self.replacement_map[ent.label_]
                masked_text = (
                    masked_text[: ent.start_char]
                    + placeholder
                    + masked_text[ent.end_char :]
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
