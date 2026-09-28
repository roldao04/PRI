"""Field extraction for structured indexing.

This module extracts different fields from Wikipedia documents:
- Title: Document title (highest weight)
- Lead: First paragraph before first section (high weight)
- Headings: Section headers (medium weight)
- Body: All other content (baseline weight)
"""

from dataclasses import dataclass


@dataclass
class DocumentFields:
    """Structured representation of document fields."""

    title: str
    lead: str
    headings: list[str]
    body: str

    def to_dict(self) -> dict[str, str | list[str]]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "lead": self.lead,
            "headings": self.headings,
            "body": self.body,
        }


class FieldExtractor:
    """Extracts structured fields from Wikipedia documents.

    This class parses Wikipedia text to identify:
    1. Title - From document metadata
    2. Lead paragraph - Opening summary before first section
    3. Section headings - Structural markers like "História", "Demografia"
    4. Body - Remaining content

    Field extraction enables field-weighted ranking where title and heading
    matches are more valuable than body matches.
    """

    def __init__(
        self,
        max_heading_length: int = 60,
        min_heading_length: int = 2,
    ):
        """Initialize field extractor.

        Args:
            max_heading_length: Maximum length for text to be considered a heading
            min_heading_length: Minimum length for text to be considered a heading
        """
        self.max_heading_length = max_heading_length
        self.min_heading_length = min_heading_length

    def extract_fields(self, title: str, text: str) -> DocumentFields:
        """Extract structured fields from document text.

        Args:
            title: Document title from metadata
            text: Full document text content

        Returns:
            DocumentFields object with extracted title, lead, headings, and body
        """
        if not text:
            return DocumentFields(title=title, lead="", headings=[], body="")

        # Split into lines for analysis
        lines = text.split("\n")

        # Extract lead paragraph (first non-empty lines before first section)
        lead_lines = []
        first_heading_idx = None

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Check if this looks like a heading
            if self._is_heading(stripped):
                first_heading_idx = i
                break

            # Add non-empty lines to lead
            if stripped:
                lead_lines.append(stripped)

        lead = " ".join(lead_lines)

        # Extract headings and body
        headings = []
        body_lines = []

        # Start from after the lead
        start_idx = first_heading_idx if first_heading_idx is not None else len(lead_lines)

        for i in range(start_idx, len(lines)):
            stripped = lines[i].strip()

            if not stripped:
                continue

            if self._is_heading(stripped):
                headings.append(stripped)
            else:
                body_lines.append(stripped)

        body = " ".join(body_lines)

        return DocumentFields(
            title=title,
            lead=lead,
            headings=headings,
            body=body,
        )

    def _is_heading(self, text: str) -> bool:
        """Determine if a line of text is likely a section heading.

        Heuristics:
        - Length between min and max heading length
        - Doesn't end with common sentence punctuation
        - Not a full URL or list item pattern

        Args:
            text: Stripped line of text

        Returns:
            True if text appears to be a heading
        """
        if not text:
            return False

        # Check length constraints
        if len(text) < self.min_heading_length or len(text) > self.max_heading_length:
            return False

        # Headings typically don't end with sentence-ending punctuation
        if text.endswith((".", ",", ":", ";")):
            return False

        # Skip URLs and links
        if text.startswith(("http://", "https://", "www.", "[")):
            return False

        # Skip lines that are clearly content (too long sentences)
        # Count spaces as rough proxy for sentence complexity
        word_count = len(text.split())
        if word_count > 10:  # Headings are typically short
            return False

        return True
