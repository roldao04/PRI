"""Document embedding generation using sentence-transformers.

This module provides functionality to generate dense vector embeddings for documents,
enabling semantic similarity search that goes beyond lexical matching. The embeddings
are precomputed once and reused for fast similarity lookups.

Recommended models for Portuguese:
- paraphrase-multilingual-mpnet-base-v2: 384 dims, best multilingual quality
- distiluse-base-multilingual-cased-v2: 512 dims, faster encoding
"""

import gc
import logging
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import psutil
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DocumentEmbedder:
    """Generate and manage document embeddings for semantic search.

    This class handles:
    - Loading sentence-transformer models (with GPU support)
    - Batch encoding of documents with progress tracking
    - Saving/loading embeddings to/from disk
    - Memory-efficient processing of large document collections
    """

    # Type annotations for attributes set in __init__
    model: SentenceTransformer
    embedding_dim: int
    model_name: str
    device: str
    batch_size: int
    normalize_embeddings: bool

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        device: str | None = None,
        batch_size: int | None = None,
        normalize_embeddings: bool = True,
    ):
        """Initialize document embedder.

        Args:
            model_name: HuggingFace model name or path (default: multilingual MPNet)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            batch_size: Batch size for encoding (None for auto-select based on device)
            normalize_embeddings: Whether to L2-normalize embeddings
                (recommended for cosine similarity)

        Raises:
            ImportError: If sentence-transformers is not installed
            RuntimeError: If model loading fails
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Auto-select batch size based on device
        if batch_size is None:
            # Larger batches for GPU (512), smaller for CPU (32) to avoid memory issues
            batch_size = 512 if device == "cuda" else 32
        self.batch_size = batch_size

        logger.info(f"Initializing document embedder: model='{model_name}', device={device}")
        logger.info(f"Batch size: {batch_size}, normalize={normalize_embeddings}")

        # Load model
        try:
            self.model = SentenceTransformer(model_name, device=device)
            # Cast to int for type safety (sentence-transformers always returns int, not None)
            self.embedding_dim = cast(int, self.model.get_sentence_embedding_dimension())
            logger.info(
                f"Model loaded successfully: {model_name} " f"(embedding_dim={self.embedding_dim})"
            )
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}")
            raise RuntimeError(
                f"Failed to load embedding model '{model_name}'. "
                "Make sure sentence-transformers is installed: "
                "pip install sentence-transformers"
            ) from e

    def encode_documents(
        self,
        documents: list[dict[str, Any]],
        text_fields: list[str] = ["title", "content"],
        max_length: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Encode documents into embeddings.

        Args:
            documents: List of document dictionaries
            text_fields: Fields to concatenate for embedding (default: ['title', 'content'])
            max_length: Maximum text length in chars (None for no truncation)
            show_progress: Whether to show progress bar

        Returns:
            numpy array of shape (num_docs, embedding_dim)
        """
        num_docs = len(documents)
        logger.info(
            f"Encoding {num_docs} documents with model '{self.model_name}' "
            f"(batch_size={self.batch_size})"
        )

        # Prepare texts
        texts = []
        for doc in documents:
            # Concatenate specified fields
            parts = []
            for field in text_fields:
                if field in doc and doc[field]:
                    parts.append(str(doc[field]))

            text = " ".join(parts)

            # Truncate if needed
            if max_length and len(text) > max_length:
                text = text[:max_length]

            texts.append(text)

        # Encode with progress bar
        start_time = time.time()

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            device=self.device,
        )

        elapsed = time.time() - start_time
        docs_per_sec = num_docs / elapsed if elapsed > 0 else 0

        logger.info(
            f"Encoding complete: {num_docs} docs in {elapsed:.2f}s "
            f"({docs_per_sec:.1f} docs/sec)"
        )

        return embeddings

    def encode_documents_from_db(
        self,
        db_path: Path,
        output_path: Path,
        text_fields: list[str] = ["title", "content"],
        max_length: int = 5000,
        save_frequency: int = 10000,
    ) -> dict[str, Any]:
        """Encode documents from SQLite database with resource-aware checkpointing.

        This method processes documents in batches using pre-allocated arrays to avoid
        memory spikes, and monitors system resources during processing.

        **NEW**: Also generates doc_id mapping file to handle non-contiguous doc_ids.
        The mapping file allows correct translation between doc_id and embedding array index.

        Args:
            db_path: Path to SQLite database with documents
            output_path: Path to save embeddings (.npy file)
            text_fields: Fields to use for embedding
            max_length: Maximum text length per document (chars)
            save_frequency: Save checkpoint every N documents

        Returns:
            Dictionary with encoding statistics
        """
        import sqlite3

        logger.info(f"Loading documents from database: {db_path}")

        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get total document count
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_docs = cursor.fetchone()[0]

        logger.info(f"Found {total_docs} documents in database")

        # Check system resources before starting
        self._check_resources(total_docs)

        # Check if partial results exist
        checkpoint_path = output_path.with_suffix(".checkpoint.npy")
        mapping_checkpoint_path = output_path.with_name(
            output_path.stem + "_mapping.checkpoint.npy"
        )
        start_idx = 0

        # PRE-ALLOCATE numpy arrays (avoids memory spike!)
        embeddings = np.zeros((total_docs, self.embedding_dim), dtype=np.float32)
        # NEW: Doc_ID mapping array (doc_id for each embedding index)
        doc_id_mapping = np.zeros(total_docs, dtype=np.int32)

        if checkpoint_path.exists():
            logger.info(f"Found checkpoint: {checkpoint_path}")
            try:
                partial = np.load(checkpoint_path)
                if partial.shape[1] == self.embedding_dim:
                    embeddings[: len(partial)] = partial
                    start_idx = len(partial)
                    logger.info(f"Resuming from document {start_idx}/{total_docs}")
                    del partial  # Free memory

                    # Load mapping checkpoint if exists
                    if mapping_checkpoint_path.exists():
                        partial_mapping = np.load(mapping_checkpoint_path)
                        doc_id_mapping[: len(partial_mapping)] = partial_mapping
                        logger.info(
                            f"Loaded doc_id mapping checkpoint: {len(partial_mapping)} entries"
                        )
                        del partial_mapping

                    gc.collect()
                else:
                    logger.warning(
                        f"Checkpoint dimension mismatch: {partial.shape[1]} != {self.embedding_dim}"
                    )
                    logger.warning("Starting fresh...")
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.warning("Starting fresh...")

        start_time = time.time()
        current_idx = start_idx
        batch_texts = []

        # Query documents
        query = """
            SELECT doc_id, title, content
            FROM documents
            ORDER BY doc_id
        """

        cursor.execute(query)

        # Skip already processed documents
        if start_idx > 0:
            for _ in range(start_idx):
                cursor.fetchone()

        logger.info("Starting encoding with resource monitoring...")
        with tqdm(total=total_docs - start_idx, desc="Encoding documents") as pbar:
            while current_idx < total_docs:
                rows = cursor.fetchmany(self.batch_size)
                if not rows:
                    break

                # Prepare batch
                batch_doc_ids = []
                for row in rows:
                    doc_id, title, content = row
                    batch_doc_ids.append(doc_id)

                    # Concatenate fields
                    parts = []
                    if "title" in text_fields and title:
                        parts.append(title)
                    if "content" in text_fields and content:
                        if len(content) > max_length:
                            content = content[:max_length]
                        parts.append(content)

                    text = " ".join(parts)
                    batch_texts.append(text)

                # Encode batch and write directly to pre-allocated array
                if batch_texts:
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        batch_size=self.batch_size,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=self.normalize_embeddings,
                        device=self.device,
                    )

                    # Write directly to arrays (no list accumulation!)
                    batch_size = len(batch_embeddings)
                    embeddings[current_idx : current_idx + batch_size] = batch_embeddings
                    # NEW: Store doc_id mapping (array_index -> doc_id)
                    doc_id_mapping[current_idx : current_idx + batch_size] = batch_doc_ids
                    current_idx += batch_size

                    # Update progress
                    pbar.update(batch_size)

                    # Save checkpoint periodically
                    if current_idx % save_frequency == 0:
                        np.save(checkpoint_path, embeddings[:current_idx])
                        np.save(mapping_checkpoint_path, doc_id_mapping[:current_idx])
                        logger.info(
                            f"Checkpoint saved: {current_idx}/{total_docs} docs (with mapping)"
                        )
                        self._log_resources()
                        gc.collect()  # Force garbage collection

                    # Monitor resources and adjust if needed
                    if current_idx % (save_frequency // 2) == 0:
                        self._monitor_and_adjust()

                    # Clear batch
                    batch_texts = []
                    batch_doc_ids = []

        # Save final embeddings and doc_id mapping
        logger.info(f"Saving final embeddings to: {output_path}")
        np.save(output_path, embeddings)

        # NEW: Save doc_id mapping file
        mapping_path = output_path.with_name(output_path.stem + "_doc_id_mapping.npy")
        logger.info(f"Saving doc_id mapping to: {mapping_path}")
        np.save(mapping_path, doc_id_mapping)

        # Verify mapping (log first few and statistics)
        unique_doc_ids = len(np.unique(doc_id_mapping))
        logger.info(
            f"Doc_ID mapping saved: {total_docs} entries, " f"{unique_doc_ids} unique doc_ids"
        )
        logger.info(f"Doc_ID range: {doc_id_mapping.min()} to {doc_id_mapping.max()}")
        logger.info(f"First 5 mappings: {list(doc_id_mapping[:5])}")

        # Remove checkpoints
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Embedding checkpoint removed")
        if mapping_checkpoint_path.exists():
            mapping_checkpoint_path.unlink()
            logger.info("Mapping checkpoint removed")

        # Close database
        conn.close()

        elapsed = time.time() - start_time
        docs_per_sec = total_docs / elapsed if elapsed > 0 else 0

        stats = {
            "num_documents": total_docs,
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "device": self.device,
            "elapsed_time_sec": elapsed,
            "docs_per_sec": docs_per_sec,
            "file_size_mb": output_path.stat().st_size / (1024 * 1024),
        }

        logger.info(
            f"Embedding generation complete: {stats['num_documents']} docs "
            f"in {stats['elapsed_time_sec']:.2f}s ({stats['docs_per_sec']:.1f} docs/sec)"
        )
        logger.info(
            f"File size: {stats['file_size_mb']:.2f} MB, "
            f"embedding dim: {stats['embedding_dim']}"
        )

        return stats

    def save_embeddings(self, embeddings: np.ndarray, output_path: Path) -> None:
        """Save embeddings to disk.

        Args:
            embeddings: numpy array of embeddings
            output_path: Path to save file (.npy format)
        """
        logger.info(f"Saving embeddings: shape={embeddings.shape} to {output_path}")
        np.save(output_path, embeddings)

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Embeddings saved: {file_size_mb:.2f} MB")

    def load_embeddings(self, input_path: Path) -> np.ndarray:
        """Load embeddings from disk.

        Args:
            input_path: Path to embeddings file (.npy format)

        Returns:
            numpy array of embeddings
        """
        logger.info(f"Loading embeddings from: {input_path}")
        embeddings = np.load(input_path)

        logger.info(
            f"Embeddings loaded: shape={embeddings.shape}, "
            f"size={input_path.stat().st_size / (1024 * 1024):.2f} MB"
        )

        return embeddings

    def _check_resources(self, total_docs: int) -> None:
        """Check if system has enough resources before starting.

        Args:
            total_docs: Total number of documents to process

        Raises:
            RuntimeError: If insufficient resources available
        """
        # Calculate required memory
        array_size_gb = (total_docs * self.embedding_dim * 4) / (1024**3)  # float32 = 4 bytes
        model_size_gb = 1.5  # Approximate model size
        buffer_gb = 0.5  # Buffer for processing
        required_gb = array_size_gb + model_size_gb + buffer_gb

        # Check available RAM
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)

        logger.info("Resource check:")
        logger.info(f"  Required memory: ~{required_gb:.2f} GB")
        logger.info(f"  Available memory: {available_gb:.2f} GB")
        logger.info(f"  Array size: {array_size_gb:.2f} GB ({total_docs:,} x {self.embedding_dim})")

        if available_gb < required_gb:
            raise RuntimeError(
                f"Insufficient memory: need {required_gb:.2f} GB but only "
                f"{available_gb:.2f} GB available. Close other applications or "
                f"use a system with more RAM."
            )

        # Check GPU memory if using CUDA
        if self.device == "cuda":
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_mem_free = torch.cuda.mem_get_info()[0] / (1024**3)
            logger.info(f"  GPU memory: {gpu_mem_free:.2f} GB / {gpu_mem_total:.2f} GB available")

        logger.info("✓ Resource check passed")

    def _log_resources(self) -> None:
        """Log current resource usage."""
        mem = psutil.virtual_memory()
        mem_used_gb = mem.used / (1024**3)
        mem_percent = mem.percent

        log_msg = f"Memory: {mem_used_gb:.2f} GB ({mem_percent:.1f}%)"

        if self.device == "cuda":
            gpu_mem_used = (
                torch.cuda.get_device_properties(0).total_memory - torch.cuda.mem_get_info()[0]
            ) / (1024**3)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_percent = (gpu_mem_used / gpu_mem_total) * 100
            log_msg += f", GPU: {gpu_mem_used:.2f} GB ({gpu_percent:.1f}%)"

        logger.info(log_msg)

    def _monitor_and_adjust(self) -> None:
        """Monitor resources and adjust processing if needed."""
        mem = psutil.virtual_memory()

        # Warn if memory usage is high
        if mem.percent > 85:
            logger.warning(f"High memory usage: {mem.percent:.1f}%")
            logger.warning("Consider closing other applications")
            gc.collect()

        # Clear CUDA cache if GPU memory is high
        if self.device == "cuda":
            gpu_mem_used = (
                torch.cuda.get_device_properties(0).total_memory - torch.cuda.mem_get_info()[0]
            ) / (1024**3)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_percent = (gpu_mem_used / gpu_mem_total) * 100

            if gpu_percent > 85:
                logger.warning(f"High GPU memory usage: {gpu_percent:.1f}%")
                torch.cuda.empty_cache()
                logger.info("Cleared CUDA cache")

        # Emergency check - stop if memory critical
        if mem.percent > 95:
            raise RuntimeError(
                f"Critical memory usage: {mem.percent:.1f}%. " "Stopping to prevent system crash."
            )

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the embedding model.

        Returns:
            Dictionary with model configuration
        """
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": self.device,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
        }
