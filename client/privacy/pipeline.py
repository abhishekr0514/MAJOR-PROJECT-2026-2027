import argparse

from client.privacy.anonymizer import PatternAnonymizer
from client.privacy.ner_masker import NERMasker


class PrivacyPipeline:
    """Integrated local privacy pipeline combining regex scrubbing and NER entity masking."""

    def __init__(self) -> None:
        """Initialize both NER masker and pattern anonymizer."""
        self.ner = NERMasker()
        self.regex = PatternAnonymizer()

    def process(self, raw_clinical_text: str) -> str:
        """Run full privacy scrubbing pipeline on raw clinical text notes.

        First applies regex rules for numerical identifiers (SSN, Phone, Email, MRN),
        followed by spaCy NER masking for names, dates, locations, and organizations.

        Args:
            raw_clinical_text: Unmasked clinical text note.

        Returns:
            Fully scrubbed text with zero PII exposure.
        """
        step1 = self.regex.scrub(raw_clinical_text)
        step2 = self.ner.mask_text(step1)
        return step2


def main() -> None:
    """CLI entrypoint for testing full privacy pipeline directly."""
    parser = argparse.ArgumentParser(
        description="Run integrated privacy pipeline on raw clinical notes."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Raw clinical text input to process.",
    )
    args = parser.parse_args()

    pipeline = PrivacyPipeline()
    result = pipeline.process(args.input)
    print("Anonymized Output:")
    print(result)


if __name__ == "__main__":
    main()
