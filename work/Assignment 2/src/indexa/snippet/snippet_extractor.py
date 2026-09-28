"""Semantic snippet extraction using neural models.

This module implements semantic snippet extraction to find the most relevant
paragraph within a document for a given query, using the same cross-encoder
model from the reranking pipeline.
"""

import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from indexa.reranker.neural_reranker import NeuralReranker

logger = logging.getLogger(__name__)


class SemanticSnippetExtractor:
    """Extract semantically relevant snippets from documents using cross-encoders.

    This class uses the same neural reranking model to score individual paragraphs
    within a document and select the most relevant one as a preview snippet.
    """

    def __init__(
        self,
        reranker: "NeuralReranker",
        min_paragraph_length: int = 50,
        max_paragraph_length: int = 500,
        max_paragraphs_to_score: int = 20,
    ):
        """Initialize semantic snippet extractor.

        Args:
            reranker: Pre-loaded NeuralReranker instance (reuses model from reranking)
            min_paragraph_length: Minimum paragraph length in characters (default: 50)
            max_paragraph_length: Maximum paragraph length in characters (default: 500)
            max_paragraphs_to_score: Maximum number of paragraphs to score per document
                (default: 20, to avoid scoring extremely long documents)
        """
        self.reranker = reranker
        self.min_paragraph_length = min_paragraph_length
        self.max_paragraph_length = max_paragraph_length
        self.max_paragraphs_to_score = max_paragraphs_to_score

    def extract_snippet_with_scores(
        self,
        query: str,
        document_content: str,
        max_snippet_length: int = 300,
    ) -> dict[str, Any]:
        """Extract the most relevant snippet and return all paragraph scores.

        Args:
            query: Search query string
            document_content: Full document text
            max_snippet_length: Maximum length of the returned snippet (default: 300)

        Returns:
            Dictionary with:
                - snippet: Most relevant paragraph/snippet
                - paragraph_scores: List of dicts with 'text' and 'score' for all paragraphs
                - best_score: Score of the best paragraph
        """
        # Split document into paragraphs
        paragraphs = self._split_into_paragraphs(document_content)

        if not paragraphs:
            # Fallback: return beginning of document
            snippet = self._truncate_text(document_content, max_snippet_length)
            return {
                "snippet": snippet,
                "paragraph_scores": [{"text": snippet, "score": 0.0}],
                "best_score": 0.0,
            }

        # Filter paragraphs by length
        valid_paragraphs = [
            p
            for p in paragraphs
            if self.min_paragraph_length <= len(p) <= self.max_paragraph_length
        ]

        if not valid_paragraphs:
            # No valid paragraphs, fallback to first paragraph or beginning
            if paragraphs:
                snippet = self._truncate_text(paragraphs[0], max_snippet_length)
            else:
                snippet = self._truncate_text(document_content, max_snippet_length)
            return {
                "snippet": snippet,
                "paragraph_scores": [{"text": snippet, "score": 0.0}],
                "best_score": 0.0,
            }

        # Limit number of paragraphs to score (performance optimization)
        if len(valid_paragraphs) > self.max_paragraphs_to_score:
            logger.debug(
                f"Too many paragraphs ({len(valid_paragraphs)}), "
                f"scoring first {self.max_paragraphs_to_score}"
            )
            valid_paragraphs = valid_paragraphs[: self.max_paragraphs_to_score]

        # Score paragraphs using the cross-encoder
        try:
            scored_paragraphs = self._score_paragraphs(query, valid_paragraphs)

            # Get best paragraph
            best_paragraph = scored_paragraphs[0]["text"]
            best_score = scored_paragraphs[0]["score"]

            # Truncate snippet if needed
            snippet = self._truncate_text(best_paragraph, max_snippet_length)

            logger.debug(f"Extracted snippet (score: {best_score:.3f}): {snippet[:100]}...")

            return {
                "snippet": snippet,
                "paragraph_scores": scored_paragraphs,
                "best_score": best_score,
            }

        except Exception as e:
            logger.error(f"Error extracting snippet: {e}")
            # Fallback to first valid paragraph
            snippet = self._truncate_text(valid_paragraphs[0], max_snippet_length)
            return {
                "snippet": snippet,
                "paragraph_scores": [{"text": snippet, "score": 0.0}],
                "best_score": 0.0,
            }

    def extract_snippet(
        self,
        query: str,
        document_content: str,
        max_snippet_length: int = 300,
    ) -> str:
        """Extract the most relevant snippet from a document for a query.

        This is a convenience method that returns only the snippet text.
        Use extract_snippet_with_scores() to also get paragraph scores.

        Args:
            query: Search query string
            document_content: Full document text
            max_snippet_length: Maximum length of the returned snippet (default: 300)

        Returns:
            Most relevant paragraph/snippet, truncated to max_snippet_length if needed
        """
        result = self.extract_snippet_with_scores(query, document_content, max_snippet_length)
        return result["snippet"]

    def extract_snippets_batch(
        self,
        query: str,
        documents: list[dict[str, Any]],
        max_snippet_length: int = 300,
    ) -> list[str]:
        """Extract snippets for multiple documents (batch processing).

        Args:
            query: Search query string
            documents: List of documents with 'content' field
            max_snippet_length: Maximum length of each snippet

        Returns:
            List of snippets (one per document, in same order)
        """
        snippets = []

        for doc in documents:
            content = doc.get("content", "")
            if not content:
                # Fallback to title or empty
                snippets.append(doc.get("title", ""))
                continue

            snippet = self.extract_snippet(query, content, max_snippet_length)
            snippets.append(snippet)

        return snippets

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs.

        Uses a multi-stage approach:
        1. Split by double newlines (standard paragraph separator)
        2. Split by single newlines (separate lines)
        3. Merge very short consecutive lines if needed
        4. Split very long paragraphs by sentences

        Args:
            text: Document text

        Returns:
            List of paragraph strings
        """
        # Stage 1: Split by double newlines first
        blocks = re.split(r"\n\n+", text)

        # Stage 2: Within each block, split by single newlines
        # This handles Wikipedia-style line breaks better
        paragraphs = []
        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Split by single newlines
            lines = block.split("\n")

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Keep all lines regardless of length to preserve document structure
                # The filtering by length happens later in extract_snippet_with_scores
                paragraphs.append(line)

        # Stage 3: Split very long paragraphs by sentences
        final_paragraphs = []
        for para in paragraphs:
            if len(para) > self.max_paragraph_length * 2:
                # Split by sentence boundaries (period/question/exclamation + space + capital)
                sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", para)

                # Group sentences into reasonable chunks
                current_chunk = []
                current_length = 0

                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue

                    # If adding this sentence would make chunk too long, save current chunk
                    if current_length + len(sent) > self.max_paragraph_length and current_chunk:
                        final_paragraphs.append(" ".join(current_chunk))
                        current_chunk = [sent]
                        current_length = len(sent)
                    else:
                        current_chunk.append(sent)
                        current_length += len(sent) + 1  # +1 for space

                # Add remaining chunk
                if current_chunk:
                    final_paragraphs.append(" ".join(current_chunk))
            else:
                final_paragraphs.append(para)

        # Stage 4: Filter out empty paragraphs and return
        return [p.strip() for p in final_paragraphs if p.strip()]

    def _score_paragraphs(
        self,
        query: str,
        paragraphs: list[str],
    ) -> list[dict[str, Any]]:
        """Score paragraphs using the cross-encoder model.

        Args:
            query: Search query
            paragraphs: List of paragraph texts

        Returns:
            List of dicts with 'text' and 'score', sorted by score (descending)
        """
        # Prepare query-paragraph pairs
        pairs = [[query, para] for para in paragraphs]

        logger.debug(f"Scoring {len(pairs)} paragraphs with cross-encoder")

        # Score all pairs
        scores = self.reranker.model.predict(
            pairs,
            batch_size=self.reranker.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        # Combine paragraphs with scores
        scored = [{"text": para, "score": float(score)} for para, score in zip(paragraphs, scores)]

        # Sort by score (descending)
        scored.sort(key=lambda x: x["score"], reverse=True)

        return scored

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to max_length with ellipsis if needed.

        Args:
            text: Text to truncate
            max_length: Maximum length in characters

        Returns:
            Truncated text with "..." appended if truncated
        """
        text = text.strip()
        if len(text) <= max_length:
            return text

        # Truncate at word boundary
        truncated = text[:max_length].rsplit(" ", 1)[0]
        return truncated + "..."
