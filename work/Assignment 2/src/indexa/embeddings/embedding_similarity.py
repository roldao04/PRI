"""Embedding-based similarity search for semantic document matching.

This module provides fast similarity search using precomputed document embeddings.
Unlike TF-IDF similarity which relies on lexical overlap, embedding-based similarity
uses dense vector representations that capture semantic meaning.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingSimilarity:
    """Fast semantic similarity search using precomputed embeddings.

    This class provides:
    - Loading of precomputed document embeddings
    - Fast cosine similarity computation using numpy
    - Top-K similarity search with efficient sorting
    - Fallback behavior when embeddings are unavailable
    """

    def __init__(self, embeddings_path: Path | None = None, auto_load: bool = True):
        """Initialize embedding similarity search.

        Args:
            embeddings_path: Path to precomputed embeddings file (.npy)
            auto_load: Whether to automatically load embeddings on init (default: True)

        Raises:
            FileNotFoundError: If embeddings_path doesn't exist and auto_load=True
        """
        self.embeddings_path = embeddings_path
        self.embeddings: np.ndarray | None = None
        self.num_documents = 0
        self.embedding_dim = 0
        self.is_loaded = False

        # NEW: Doc_ID mapping support (handles non-contiguous doc_ids)
        self.doc_id_mapping: np.ndarray | None = None  # array_index -> doc_id
        self.doc_id_to_index: dict[int, int] | None = None  # doc_id -> array_index (reverse)
        self.has_mapping = False

        if auto_load and embeddings_path is not None:
            self.load_embeddings(embeddings_path)

    def load_embeddings(self, embeddings_path: Path) -> None:
        """Load precomputed embeddings from disk.

        **NEW**: Also loads doc_id mapping file to handle non-contiguous doc_ids.
        The mapping file should be named: {embeddings_name}_doc_id_mapping.npy

        Args:
            embeddings_path: Path to embeddings file (.npy format)

        Raises:
            FileNotFoundError: If embeddings file doesn't exist
            ValueError: If embeddings have invalid shape
        """
        if not embeddings_path.exists():
            raise FileNotFoundError(
                f"Embeddings file not found: {embeddings_path}\n"
                "Generate embeddings using: python scripts/generate_embeddings.py"
            )

        logger.info(f"Loading embeddings from: {embeddings_path}")
        start_time = None
        try:
            import time

            start_time = time.time()
            self.embeddings = np.load(embeddings_path)
            assert self.embeddings is not None  # Type checker hint

            # Validate shape
            if len(self.embeddings.shape) != 2:
                raise ValueError(
                    f"Invalid embeddings shape: {self.embeddings.shape}. "
                    "Expected 2D array (num_docs, embedding_dim)"
                )

            self.num_documents, self.embedding_dim = self.embeddings.shape
            self.embeddings_path = embeddings_path
            self.is_loaded = True

            elapsed = time.time() - start_time if start_time else 0
            file_size_mb = embeddings_path.stat().st_size / (1024 * 1024)

            logger.info(
                f"Embeddings loaded successfully: "
                f"{self.num_documents:,} docs, dim={self.embedding_dim}, "
                f"size={file_size_mb:.2f} MB, load_time={elapsed:.2f}s"
            )

            # NEW: Try to load doc_id mapping file
            mapping_path = embeddings_path.with_name(embeddings_path.stem + "_doc_id_mapping.npy")
            if mapping_path.exists():
                logger.info(f"Loading doc_id mapping from: {mapping_path}")
                self.doc_id_mapping = np.load(mapping_path)
                assert self.doc_id_mapping is not None, "Doc ID mapping must not be None"

                # Build reverse mapping (doc_id -> array_index) for O(1) lookup
                self.doc_id_to_index = {
                    int(doc_id): idx for idx, doc_id in enumerate(self.doc_id_mapping)
                }
                self.has_mapping = True

                mapping_file_size = mapping_path.stat().st_size / (1024 * 1024)
                logger.info(
                    f"Doc_ID mapping loaded: {len(self.doc_id_mapping):,} entries, "
                    f"size={mapping_file_size:.2f} MB"
                )
                logger.info(
                    f"Doc_ID range: {self.doc_id_mapping.min()} to {self.doc_id_mapping.max()}"
                )
            else:
                logger.warning(
                    f"Doc_ID mapping file not found: {mapping_path}\n"
                    "Assuming doc_ids are contiguous (0 to N-1). "
                    "If you have non-contiguous doc_ids, "
                    "regenerate embeddings with the latest code."
                )
                self.has_mapping = False

        except Exception as e:
            logger.error(f"Failed to load embeddings from {embeddings_path}: {e}")
            raise

    def _get_embedding_index(self, doc_id: int) -> int:
        """Translate doc_id to embedding array index.

        This method handles the mapping between actual doc_ids and array indices,
        supporting non-contiguous doc_ids.

        Args:
            doc_id: Actual document ID from the index

        Returns:
            Array index for embeddings array

        Raises:
            ValueError: If doc_id is not found in mapping
        """
        if self.has_mapping and self.doc_id_to_index is not None:
            # Use mapping to translate doc_id to array index
            if doc_id not in self.doc_id_to_index:
                available_range = (
                    f"{self.doc_id_mapping.min()} to {self.doc_id_mapping.max()}"
                    if self.doc_id_mapping is not None
                    else "unknown"
                )
                raise ValueError(
                    f"Doc_ID {doc_id} not found in embeddings mapping. "
                    f"Available doc_id range: {available_range}. "
                    f"This document may not have been included during embedding generation."
                )
            return self.doc_id_to_index[doc_id]
        else:
            # No mapping - assume contiguous doc_ids (0 to N-1)
            if doc_id < 0 or doc_id >= self.num_documents:
                raise ValueError(
                    f"Invalid doc_id: {doc_id}. Must be in range [0, {self.num_documents}). "
                    f"If your doc_ids are non-contiguous, "
                    f"regenerate embeddings with the latest code to create a doc_id mapping file."
                )
            return doc_id

    def find_similar(
        self,
        doc_id: int,
        top_k: int = 10,
        min_similarity: float = 0.0,
        include_scores: bool = True,
    ) -> list[tuple[int, float]] | list[int]:
        """Find most similar documents using embedding cosine similarity.

        **NEW**: Now supports non-contiguous doc_ids via mapping file.

        Args:
            doc_id: Source document ID (actual doc_id from index)
            top_k: Number of similar documents to return
            min_similarity: Minimum cosine similarity threshold (0.0 to 1.0)
            include_scores: Whether to return similarity scores with doc IDs

        Returns:
            List of (doc_id, similarity_score) tuples if include_scores=True,
            otherwise list of doc_ids only

        Raises:
            ValueError: If embeddings are not loaded or doc_id is invalid
        """
        if not self.is_loaded or self.embeddings is None:
            raise ValueError(
                "Embeddings not loaded. Call load_embeddings() first or "
                "set auto_load=True in constructor."
            )

        # NEW: Translate doc_id to embedding array index
        try:
            embedding_idx = self._get_embedding_index(doc_id)
        except ValueError as e:
            # Re-raise with more context
            logger.error(f"Failed to find embedding for doc_id={doc_id}: {e}")
            raise

        # Get query embedding (ensure it's 1D)
        query_embedding = self.embeddings[embedding_idx]

        # Compute cosine similarity with all documents
        # Note: Embeddings are assumed to be L2-normalized, so dot product = cosine similarity
        similarities = np.dot(self.embeddings, query_embedding)

        # Set query document similarity to -1 to exclude it from results
        similarities[embedding_idx] = -1.0

        # Filter by minimum similarity
        if min_similarity > 0:
            valid_mask = similarities >= min_similarity
            valid_indices = np.where(valid_mask)[0]
            valid_similarities = similarities[valid_indices]

            # Sort by similarity (descending)
            sorted_idx = np.argsort(-valid_similarities)
            top_indices = valid_indices[sorted_idx[:top_k]]
            top_similarities = valid_similarities[sorted_idx[:top_k]]
        else:
            # Get top-k without filtering
            # Use argpartition for efficiency (O(n) instead of O(n log n))
            if top_k < self.num_documents:
                top_indices_unsorted = np.argpartition(-similarities, top_k)[:top_k]
                # Sort just the top-k (much faster than sorting all)
                sorted_within_top = np.argsort(-similarities[top_indices_unsorted])
                top_indices = top_indices_unsorted[sorted_within_top]
                top_similarities = similarities[top_indices]
            else:
                # Return all, sorted
                sorted_idx = np.argsort(-similarities)
                top_indices = sorted_idx[:top_k]
                top_similarities = similarities[top_indices]

        # NEW: Convert array indices back to doc_ids
        if self.has_mapping and self.doc_id_mapping is not None:
            # Use mapping to convert indices to actual doc_ids
            doc_ids = [int(self.doc_id_mapping[idx]) for idx in top_indices]
        else:
            # No mapping - indices are doc_ids
            doc_ids = [int(idx) for idx in top_indices]

        # Return results
        if include_scores:
            results = [(doc_id, float(sim)) for doc_id, sim in zip(doc_ids, top_similarities)]
        else:
            results = doc_ids

        logger.debug(
            f"Found {len(results)} similar documents for doc_id={doc_id} "
            f"(top similarity: {top_similarities[0]:.3f})"
            if len(results) > 0
            else f"No similar documents found for doc_id={doc_id}"
        )

        return results

    def batch_find_similar(
        self,
        doc_ids: list[int],
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> dict[int, list[tuple[int, float]]]:
        """Find similar documents for multiple source documents (batched).

        Args:
            doc_ids: List of source document IDs
            top_k: Number of similar documents per source
            min_similarity: Minimum cosine similarity threshold

        Returns:
            Dictionary mapping source doc_id to list of (similar_doc_id, score) tuples

        Raises:
            ValueError: If embeddings are not loaded
        """
        if not self.is_loaded or self.embeddings is None:
            raise ValueError("Embeddings not loaded")

        results = {}
        for doc_id in doc_ids:
            try:
                results[doc_id] = self.find_similar(
                    doc_id=doc_id,
                    top_k=top_k,
                    min_similarity=min_similarity,
                    include_scores=True,
                )
            except Exception as e:
                logger.error(f"Error finding similar docs for {doc_id}: {e}")
                results[doc_id] = []

        return results

    def compute_similarity_matrix(self, doc_ids: list[int]) -> tuple[np.ndarray, list[int]]:
        """Compute pairwise similarity matrix for a set of documents.

        Args:
            doc_ids: List of document IDs

        Returns:
            Tuple of (similarity_matrix, doc_ids) where similarity_matrix[i, j]
            is the cosine similarity between doc_ids[i] and doc_ids[j]

        Raises:
            ValueError: If embeddings are not loaded
        """
        if not self.is_loaded or self.embeddings is None:
            raise ValueError("Embeddings not loaded")

        # Get embeddings for specified documents
        doc_embeddings = self.embeddings[doc_ids]

        # Compute pairwise similarities (matrix multiplication)
        similarity_matrix = np.dot(doc_embeddings, doc_embeddings.T)

        return similarity_matrix, doc_ids

    def get_info(self) -> dict[str, Any]:
        """Get information about loaded embeddings.

        Returns:
            Dictionary with embedding statistics
        """
        info = {
            "is_loaded": self.is_loaded,
            "num_documents": self.num_documents,
            "embedding_dim": self.embedding_dim,
            "embeddings_path": str(self.embeddings_path) if self.embeddings_path else None,
            "memory_usage_mb": (
                self.embeddings.nbytes / (1024 * 1024) if self.embeddings is not None else 0
            ),
            "has_mapping": self.has_mapping,
        }

        # Add doc_id range if mapping exists
        if self.has_mapping and self.doc_id_mapping is not None:
            info["doc_id_range"] = {
                "min": int(self.doc_id_mapping.min()),
                "max": int(self.doc_id_mapping.max()),
                "count": len(self.doc_id_mapping),
            }

        return info

    def is_available(self) -> bool:
        """Check if embeddings are loaded and ready to use.

        Returns:
            True if embeddings are loaded, False otherwise
        """
        return self.is_loaded and self.embeddings is not None

    def unload(self) -> None:
        """Unload embeddings from memory to free resources."""
        if self.embeddings is not None:
            logger.info("Unloading embeddings from memory")
            self.embeddings = None
            self.is_loaded = False
            self.num_documents = 0
            self.embedding_dim = 0
