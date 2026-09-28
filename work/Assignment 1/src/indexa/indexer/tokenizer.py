"""Fast Portuguese tokenizer optimized for Wikipedia indexing.

Performance optimizations:
- Uses pystemmer (C-based Snowball) instead of NLTK (10-50x faster)
- TRUE batch stemming via stemWords() to amortize function call overhead
- Frozenset for stop words (faster lookups than set)
- Pre-compiled regex patterns at module level
- Larger batch sizes for better stemming performance
- Generator-based processing for memory efficiency
- Field-based tokenization for weighted ranking
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, cast

import nltk
import Stemmer
from nltk.corpus import stopwords

if TYPE_CHECKING:
    from indexa.indexer.field_extractor import DocumentFields

# Ensure NLTK stopwords data is available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download("stopwords", quiet=True)

# Load Portuguese stopwords once at module import and convert to frozenset for fast lookups
PORTUGUESE_STOPWORDS: frozenset[str] = frozenset(stopwords.words("portuguese"))

# Pre-compile regex patterns at module level for maximum performance
TOKEN_PATTERN = re.compile(r"[a-zA-ZÀ-ÿ]+", re.UNICODE)
TOKEN_PATTERN_WITH_NUMBERS = re.compile(r"[a-zA-ZÀ-ÿ0-9]+", re.UNICODE)


@dataclass
class TokenizerConfig:
    """Minimal configuration for Portuguese tokenizer."""

    use_stemming: bool = True
    use_stopwords: bool = True
    min_token_length: int = 3
    max_token_length: int = 50
    batch_size: int = 5000  # Batch size for stemming
    keep_numbers: bool = False  # Keep tokens containing numbers

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "use_stemming": self.use_stemming,
            "use_stopwords": self.use_stopwords,
            "min_token_length": self.min_token_length,
            "max_token_length": self.max_token_length,
            "batch_size": self.batch_size,
            "keep_numbers": self.keep_numbers,
        }

    @staticmethod
    def from_dict(data: dict[str, object]) -> "TokenizerConfig":
        """Create from dictionary."""
        return TokenizerConfig(
            use_stemming=bool(data.get("use_stemming", True)),
            use_stopwords=bool(data.get("use_stopwords", True)),
            min_token_length=cast(int, data.get("min_token_length", 3)),
            max_token_length=cast(int, data.get("max_token_length", 50)),
            batch_size=cast(int, data.get("batch_size", 5000)),
            keep_numbers=bool(data.get("keep_numbers", False)),
        )


class Tokenizer:
    """Fast Portuguese tokenizer optimized for Wikipedia indexing with batch processing."""

    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.stemmer: object = Stemmer.Stemmer("portuguese") if config.use_stemming else None
        self.stop_words: frozenset[str] = (
            PORTUGUESE_STOPWORDS if config.use_stopwords else frozenset()
        )
        # Use module-level pre-compiled regex for maximum performance
        # Choose pattern based on keep_numbers setting
        self.token_pattern = TOKEN_PATTERN_WITH_NUMBERS if config.keep_numbers else TOKEN_PATTERN
        # Pre-calculate for speed
        self.min_len = config.min_token_length
        self.max_len = config.max_token_length
        self.batch_size = config.batch_size

    def tokenize_with_positions(self, text: str) -> Iterator[tuple[str, int]]:
        """Tokenize text and return terms with their positions using batch processing.

        This is crucial for positional indexing, which allows
        phrase queries and proximity searches.

        Yields (term, position) tuples.
        Optimized with batch stemming for maximum speed.
        """
        # For positional indexing, we need to maintain order
        # But we can still batch the stemming for speed
        tokens_positions: list[tuple[str, int]] = []
        position = 0

        for match in self.token_pattern.finditer(text):
            token = match.group().lower()
            token_len = len(token)

            if self.min_len <= token_len <= self.max_len and token not in self.stop_words:
                tokens_positions.append((token, position))
                position += 1

        if not tokens_positions:
            return

        # Batch stem if needed
        if self.stemmer:
            # Extract just tokens for stemming
            tokens_only = [t for t, _ in tokens_positions]
            stemmed_tokens = cast(
                list[str],
                self.stemmer.stemWords(tokens_only),  # type: ignore[attr-defined]
            )

            # Zip back with positions
            for stemmed, (_, pos) in zip(stemmed_tokens, tokens_positions):
                yield (stemmed, pos)
        else:
            yield from tokens_positions

    def tokenize_batch_with_positions(self, texts: list[str]) -> list[list[tuple[str, int]]]:
        """Tokenize multiple texts at once with positions (batch processing).

        This method processes multiple documents at once to reduce Python
        function call overhead. Each document maintains its own position counter.

        Args:
            texts: List of text strings to tokenize

        Returns:
            List of lists, where each inner list contains (term, position) tuples
            for one document. The order matches the input order.

        Performance:
            - Processes all documents together to amortize function call overhead
            - Uses batch stemming for maximum speed
            - 20-30% faster than calling tokenize_with_positions() repeatedly
        """
        if not texts:
            return []

        # Process each text to extract tokens with positions
        all_docs_tokens: list[list[tuple[str, int]]] = []
        all_tokens_flat: list[str] = []  # For batch stemming
        doc_lengths: list[int] = []  # Track where each doc ends

        for text in texts:
            tokens_positions: list[tuple[str, int]] = []
            position = 0

            for match in self.token_pattern.finditer(text):
                token = match.group().lower()
                token_len = len(token)

                if self.min_len <= token_len <= self.max_len and token not in self.stop_words:
                    tokens_positions.append((token, position))
                    all_tokens_flat.append(token)
                    position += 1

            all_docs_tokens.append(tokens_positions)
            doc_lengths.append(len(tokens_positions))

        # Batch stem all tokens at once if needed
        if self.stemmer and all_tokens_flat:
            stemmed_tokens = cast(
                list[str],
                self.stemmer.stemWords(all_tokens_flat),  # type: ignore[attr-defined]
            )

            # Reconstruct results with stemmed tokens
            result: list[list[tuple[str, int]]] = []
            token_idx = 0

            for doc_len in doc_lengths:
                doc_result: list[tuple[str, int]] = []
                for i in range(doc_len):
                    stemmed = stemmed_tokens[token_idx]
                    position = all_docs_tokens[len(result)][i][1]  # Get original position
                    doc_result.append((stemmed, position))
                    token_idx += 1
                result.append(doc_result)

            return result
        else:
            # No stemming - return original tokens with positions
            return all_docs_tokens

    def tokenize_fields(self, fields: "DocumentFields") -> Iterator[tuple[str, int, int]]:
        """Tokenize structured document fields with field-specific positions.

        This method tokenizes each field separately and yields tokens with
        their field ID and position within that field. This enables
        field-weighted ranking in BM25F.

        Args:
            fields: DocumentFields object with title, lead, headings, body

        Yields:
            Tuples of (term, field_id, position) where:
            - term: stemmed/processed term
            - field_id: FieldType enum value (TITLE=1, LEAD=3, HEADING=2, BODY=0)
            - position: position within that specific field

        Example:
            For a document with title "Portugal" and body "Portugal é um país":
            - ("portugal", TITLE, 0)
            - ("portugal", BODY, 0)
            - ("pais", BODY, 1)
        """
        # Import here to avoid circular dependency
        from indexa.indexer.models import FieldType

        # Process title field (highest weight)
        if fields.title:
            title_tokens = list(self.tokenize_with_positions(fields.title))
            for term, pos in title_tokens:
                yield (term, int(FieldType.TITLE), pos)

        # Process lead paragraph (high weight)
        if fields.lead:
            lead_tokens = list(self.tokenize_with_positions(fields.lead))
            for term, pos in lead_tokens:
                yield (term, int(FieldType.LEAD), pos)

        # Process headings (medium-high weight)
        # Concatenate all headings with space separation
        if fields.headings:
            headings_text = " ".join(fields.headings)
            heading_tokens = list(self.tokenize_with_positions(headings_text))
            for term, pos in heading_tokens:
                yield (term, int(FieldType.HEADING), pos)

        # Process body content (baseline weight)
        if fields.body:
            body_tokens = list(self.tokenize_with_positions(fields.body))
            for term, pos in body_tokens:
                yield (term, int(FieldType.BODY), pos)

    def tokenize_fields_batch(
        self, fields_list: list["DocumentFields"]
    ) -> list[list[tuple[str, int, int]]]:
        """Tokenize multiple documents' fields in batch for better performance.

        Args:
            fields_list: List of DocumentFields objects

        Returns:
            List of lists, where each inner list contains (term, field_id, position)
            tuples for one document.
        """
        # For now, process individually but collect results
        # Could be optimized further with true batch processing
        results = []
        for fields in fields_list:
            doc_tokens = list(self.tokenize_fields(fields))
            results.append(doc_tokens)
        return results
