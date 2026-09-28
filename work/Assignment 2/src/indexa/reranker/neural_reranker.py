"""Neural reranking using cross-encoder models.

This module implements neural reranking to improve the quality of BM25 search results
by rescoring candidate documents using transformer-based cross-encoder models.
"""

import logging
from typing import Any

import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Supported cross-encoder models
SUPPORTED_MODELS = {
    "unicamp-dl": "unicamp-dl/mMiniLM-L6-v2-pt-v2",  # Portuguese-specific, lightweight
    "mmarco": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  # Multilingual, heavier
}


class NeuralReranker:
    """Neural reranker using cross-encoder models.

    This class implements a two-stage retrieval pipeline:
    1. BM25 retrieves top-K candidates (fast but lexical)
    2. Cross-encoder reranks candidates (slow but semantic)

    The cross-encoder computes a relevance score for each query-document pair
    by processing them together (not separately like bi-encoders).
    """

    def __init__(
        self,
        model_name: str = "unicamp-dl",
        batch_size: int | None = None,
        device: str | None = None,
    ):
        """Initialize the neural reranker.

        Args:
            model_name: Model identifier from SUPPORTED_MODELS or HuggingFace model path
            batch_size: Number of query-document pairs to process at once.
                       If None, automatically selects: GPU=32, CPU=8 (optimized defaults)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        # Resolve model name
        if model_name in SUPPORTED_MODELS:
            self.model_path = SUPPORTED_MODELS[model_name]
            self.model_name = model_name
        else:
            self.model_path = model_name
            self.model_name = "custom"

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Auto-select optimal batch size based on device
        if batch_size is None:
            batch_size = 32 if device == "cuda" else 8
            logger.info(f"Auto-selected batch size: {batch_size} (device: {device})")
        self.batch_size = batch_size

        logger.info(f"Loading cross-encoder model: {self.model_path}")
        logger.info(f"Using device: {self.device}")

        # Load cross-encoder model
        try:
            self.model = CrossEncoder(self.model_path, device=device)
            logger.info(f"Model loaded successfully: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_path}: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank documents using the cross-encoder model.

        Args:
            query: Search query string
            documents: List of documents with at least 'content' or 'title' field
            top_k: Number of top results to return (None = return all, sorted)

        Returns:
            List of documents sorted by neural score (descending), with added 'neural_score' field
        """
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            # Use title + content if available, otherwise just content or title
            doc_text = self._get_document_text(doc)
            pairs.append([query, doc_text])

        logger.debug(f"Scoring {len(pairs)} query-document pairs with cross-encoder")

        # Score all pairs in batches
        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as e:
            logger.error(f"Error during cross-encoder scoring: {e}")
            # Return original documents with zero scores as fallback
            for doc in documents:
                doc["neural_score"] = 0.0
            return documents

        # Add neural scores to documents
        for doc, score in zip(documents, scores):
            doc["neural_score"] = float(score)

        # Sort by neural score (descending)
        reranked = sorted(documents, key=lambda x: x["neural_score"], reverse=True)

        # Limit to top-k if specified
        if top_k is not None:
            reranked = reranked[:top_k]

        logger.debug(
            f"Reranking complete. Score range: [{min(scores):.3f}, {max(scores):.3f}]"
            if len(scores) > 0
            else "No scores computed"
        )

        return reranked

    def _get_document_text(self, doc: dict[str, Any]) -> str:
        """Extract text from document for scoring.

        Args:
            doc: Document dictionary

        Returns:
            Text representation of the document
        """
        # Priority: title + content > content > title
        title = doc.get("title", "")
        content = doc.get("content", "")

        if title and content:
            # Truncate content to avoid exceeding model's max length
            # Most cross-encoders support 512 tokens, truncate to ~200 words
            # Shorter text = faster inference, and most relevant info is at the beginning
            content_words = content.split()[:200]
            truncated_content = " ".join(content_words)
            return f"{title}. {truncated_content}"
        elif content:
            content_words = content.split()[:200]
            return " ".join(content_words)
        elif title:
            return title
        else:
            return ""

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "batch_size": self.batch_size,
        }
