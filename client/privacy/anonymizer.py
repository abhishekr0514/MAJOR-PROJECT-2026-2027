import argparse
import re


class PatternAnonymizer:
    """Regex pattern matcher for scrubbing structured PII (SSN, Phone, Email, MRN) from text."""

    def __init__(self) -> None:
        """Initialize regex pattern matchers and token replacements."""
        self.patterns: list[tuple[re.Pattern[str], str]] = [
            (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
            (
                re.compile(
                    r"(\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\(\d{3}\)\s*\d{3}[-.\s]?\d{4}|\b\d{10}\b)"
                ),
                "[PHONE]",
            ),
            (
                re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
                "[EMAIL]",
            ),
            (re.compile(r"\bMRN-?\d{6,8}\b", re.IGNORECASE), "[MRN]"),
        ]

    def scrub(self, text: str) -> str:
        """Apply regex pattern replacements to scrub numeric and identifier PII.

        Args:
            text: Raw input text.

        Returns:
            Scrubbed text with matching PII replaced by placeholders.
        """
        scrubbed = text
        for pattern, replacement in self.patterns:
            scrubbed = pattern.sub(replacement, scrubbed)
        return scrubbed


def main() -> None:
    """CLI entrypoint for testing pattern anonymizer directly."""
    parser = argparse.ArgumentParser(
        description="Scrub structured PII (SSN, Phone, Email, MRN) using regex patterns."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Raw text input to scrub.",
    )
    args = parser.parse_args()

    anonymizer = PatternAnonymizer()
    result = anonymizer.scrub(args.input)
    print("Scrubbed Output:")
    print(result)


if __name__ == "__main__":
    main()
