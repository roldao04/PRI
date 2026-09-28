#!/usr/bin/env python3
"""Batch embedding generation script for Wikipedia documents.

This script generates dense vector embeddings for all documents in the index,
enabling semantic similarity search. The embeddings are saved to a numpy file
for fast loading and reuse.

Usage:
    python scripts/generate_embeddings.py [--index-dir INDEX_DIR] [--model MODEL_NAME]

Example:
    # Use default model (multilingual MPNet, 384 dims)
    python scripts/generate_embeddings.py

    # Use specific model and index
    python scripts/generate_embeddings.py --index-dir wiki_index_full --model paraphrase-multilingual-mpnet-base-v2

    # Test with small subset first
    python scripts/generate_embeddings.py --max-docs 10000

GPU Recommendations:
    - RTX 3060/4060 (8GB): batch_size=512, ~2-3 hours for 1.15M docs
    - RTX 3080/4080 (10-16GB): batch_size=1024, ~1-2 hours
    - CPU only: batch_size=32, ~15-20 hours (not recommended for full dataset)

The script supports checkpointing - if interrupted, it will resume from where it stopped.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from indexa.embeddings.document_embedder import DocumentEmbedder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate embeddings for Wikipedia documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("wiki_index_full"),
        help="Directory containing the index and documents.db (default: wiki_index_full)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="paraphrase-multilingual-mpnet-base-v2",
        help="Sentence-transformer model name (default: paraphrase-multilingual-mpnet-base-v2)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for encoding (default: auto-select based on device)",
    )

    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to process (for testing, default: all)",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=5000,
        help="Maximum text length per document in characters (default: 5000)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for embeddings file (default: INDEX_DIR/embeddings.npy)",
    )

    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU usage even if GPU is available (not recommended)",
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization of embeddings (default: normalize for cosine similarity)",
    )

    return parser.parse_args()


def print_system_info():
    """Print system information for diagnostics."""
    print("=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)

    # GPU info
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(
            f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
        print(
            f"  Available memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB / "
            f"{torch.cuda.mem_get_info()[1] / 1e9:.2f} GB"
        )
    else:
        print("✗ CUDA not available - will use CPU (this will be slow!)")

    print(f"PyTorch version: {torch.__version__}")
    print("=" * 80 + "\n")


def estimate_time(num_docs: int, device: str, batch_size: int) -> tuple[float, float]:
    """Estimate processing time based on system specs.

    Args:
        num_docs: Number of documents to process
        device: 'cuda' or 'cpu'
        batch_size: Batch size for processing

    Returns:
        Tuple of (min_hours, max_hours)
    """
    if device == "cuda":
        # GPU: ~500-1000 docs/sec depending on GPU
        docs_per_sec = 600  # Conservative estimate for RTX 3060/4060
        if batch_size >= 1024:
            docs_per_sec = 800  # Faster GPU
    else:
        # CPU: ~10-20 docs/sec
        docs_per_sec = 15

    total_seconds = num_docs / docs_per_sec
    hours = total_seconds / 3600

    # Add 20% margin for overhead
    min_hours = hours * 0.9
    max_hours = hours * 1.3

    return min_hours, max_hours


def main():
    """Main entry point for embedding generation."""
    args = parse_args()

    # Print system info
    print_system_info()

    # Validate inputs
    db_path = args.index_dir / "documents.db"
    if not db_path.exists():
        logger.error(f"Database not found: {db_path}")
        logger.error("Make sure you're pointing to a valid index directory.")
        sys.exit(1)

    # Set output path
    if args.output is None:
        output_path = args.index_dir / "embeddings.npy"
    else:
        output_path = args.output

    # Determine device
    if args.cpu_only:
        device = "cpu"
        logger.warning("CPU-only mode enabled - this will be slow!")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Show configuration
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Index directory: {args.index_dir}")
    print(f"Database: {db_path}")
    print(f"Output file: {output_path}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size or 'auto'}")
    print(f"Max text length: {args.max_length} chars")
    print(f"Normalize embeddings: {not args.no_normalize}")
    if args.max_docs:
        print(f"Max documents: {args.max_docs} (testing mode)")
    print("=" * 80 + "\n")

    # Get document count for ETA
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM documents")
    total_docs = cursor.fetchone()[0]
    conn.close()

    if args.max_docs:
        total_docs = min(total_docs, args.max_docs)

    # Estimate time
    batch_size_for_estimate = args.batch_size or (512 if device == "cuda" else 32)
    min_hours, max_hours = estimate_time(total_docs, device, batch_size_for_estimate)

    print("ESTIMATED TIME")
    print("=" * 80)
    print(f"Documents to process: {total_docs:,}")
    if device == "cuda":
        print(f"Estimated time: {min_hours:.1f} - {max_hours:.1f} hours (GPU)")
        print(f"  (approximately {total_docs / (min_hours * 3600):.0f} docs/sec)")
    else:
        print(f"Estimated time: {min_hours:.1f} - {max_hours:.1f} hours (CPU)")
        print("  WARNING: CPU processing is very slow. Consider using GPU or Colab.")
    print("=" * 80 + "\n")

    # Confirm before proceeding
    try:
        response = input("Proceed with embedding generation? [y/N]: ").strip().lower()
        if response not in ["y", "yes"]:
            print("Cancelled by user.")
            sys.exit(0)
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled by user.")
        sys.exit(0)

    print("\nStarting embedding generation...\n")

    # Initialize embedder
    try:
        embedder = DocumentEmbedder(
            model_name=args.model,
            device=device,
            batch_size=args.batch_size,
            normalize_embeddings=not args.no_normalize,
        )
    except Exception as e:
        logger.error(f"Failed to initialize embedder: {e}")
        logger.error("Make sure sentence-transformers is installed:")
        logger.error("  pip install sentence-transformers")
        sys.exit(1)

    # Generate embeddings
    try:
        start_time = time.time()

        stats = embedder.encode_documents_from_db(
            db_path=db_path,
            output_path=output_path,
            text_fields=["title", "content"],
            max_length=args.max_length,
            save_frequency=10000,  # Save checkpoint every 10k docs
        )

        elapsed = time.time() - start_time

        # Print summary
        print("\n" + "=" * 80)
        print("EMBEDDING GENERATION COMPLETE")
        print("=" * 80)
        print(f"Documents processed: {stats['num_documents']:,}")
        print(f"Embedding dimension: {stats['embedding_dim']}")
        print(f"Model: {stats['model_name']}")
        print(f"Device: {stats['device']}")
        print(f"Total time: {elapsed / 3600:.2f} hours ({elapsed:.0f} seconds)")
        print(f"Speed: {stats['docs_per_sec']:.1f} docs/sec")
        print(f"Output file: {output_path}")
        print(f"File size: {stats['file_size_mb']:.2f} MB")
        print("=" * 80)

        print("\n✓ Success! Embeddings are ready to use.")
        print(
            f"\nTo use embeddings in your search engine, make sure '{output_path}' "
            "is in your index directory."
        )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress has been checkpointed.")
        print("Run the script again to resume from where it stopped.")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
