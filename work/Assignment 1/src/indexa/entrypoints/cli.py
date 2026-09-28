"""Command-line interface for the SPIMI indexer.

This CLI provides the main entry point for building inverted indices
with configurable tokenization and memory constraints.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from indexa.core.limit_memory import start_memory_monitor
from indexa.core.logging import setup_logging
from indexa.indexer.spimi import SPIMIIndexer
from indexa.indexer.tokenizer import Tokenizer, TokenizerConfig

# Initialize logging first
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# Start memory monitoring (2GB limit)
start_memory_monitor(show_memory_updates=True)


def parse_arguments():
    """Parse command-line arguments for the indexer."""
    parser = argparse.ArgumentParser(
        description="SPIMI Indexer CLI - Build inverted indices with memory constraints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a directory of documents with default settings
  %(prog)s /path/to/documents -o /path/to/index

  # Index with Portuguese tokenizer and custom settings
  %(prog)s /path/to/documents -o /path/to/index --min-token 3

  # Index with custom tokenizer configuration
  %(prog)s /path/to/documents -o /path/to/index --tokenizer-config config.json

  # Index with minimum term frequency filtering
  %(prog)s /path/to/documents -o /path/to/index --min-term-freq 5

Memory Constraint:
  The indexer automatically monitors memory usage and will terminate
  if it exceeds 2GB. The SPIMI algorithm is designed to work within
  this constraint by writing partial indices to disk periodically.
        """,
    )

    # Required arguments
    parser.add_argument("input_path", type=str, help="Path to documents directory or file list")

    parser.add_argument(
        "-o", "--output", type=str, required=True, help="Output directory for the index"
    )

    # Tokenizer configuration
    tokenizer_group = parser.add_argument_group("Tokenizer Configuration")

    tokenizer_group.add_argument(
        "--tokenizer-config", type=str, help="Path to tokenizer configuration JSON file"
    )

    tokenizer_group.add_argument(
        "--language",
        type=str,
        choices=["pt", "portuguese"],
        default="portuguese",
        help="Language for tokenization (default: portuguese)",
    )

    tokenizer_group.add_argument("--no-stemming", action="store_true", help="Disable stemming")

    tokenizer_group.add_argument(
        "--no-stopwords", action="store_true", help="Disable stop word removal"
    )

    tokenizer_group.add_argument(
        "--min-token", type=int, default=3, help="Minimum token length (default: 3)"
    )

    tokenizer_group.add_argument(
        "--max-token", type=int, default=50, help="Maximum token length (default: 50)"
    )

    tokenizer_group.add_argument(
        "--keep-numbers",
        action="store_true",
        help="Keep tokens containing numbers (e.g., '2023', 'covid19')",
    )

    # SPIMI configuration
    spimi_group = parser.add_argument_group("SPIMI Configuration")

    spimi_group.add_argument(
        "--memory-limit",
        type=int,
        default=1800,
        help="Memory limit in MB (default: 1800, max: 2000)",
    )

    spimi_group.add_argument(
        "--block-size",
        type=int,
        default=120,
        help="Block size threshold in MB (default: 120, reduced for field-based indexing)",
    )

    spimi_group.add_argument(
        "--batch-size", type=int, default=1000, help="Number of documents per batch (default: 1000)"
    )

    spimi_group.add_argument(
        "--min-term-freq", type=int, default=1, help="Minimum term frequency for index (default: 1)"
    )

    # Document processing
    doc_group = parser.add_argument_group("Document Processing")

    doc_group.add_argument(
        "--file-pattern",
        type=str,
        default="*.txt",
        help="File pattern for documents (default: *.txt)",
    )

    doc_group.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for documents in subdirectories",
    )

    doc_group.add_argument(
        "--max-docs", type=int, help="Maximum number of documents to index (for testing)"
    )

    # Other options
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--no-memory-monitor", action="store_true", help="Disable memory monitoring output"
    )

    return parser.parse_args()


def create_tokenizer(args: argparse.Namespace) -> Tokenizer:
    """Create tokenizer from command-line arguments.

    This function handles the various ways to configure the tokenizer:
    1. From a configuration file
    2. From command-line arguments
    3. With default settings
    """
    if args.tokenizer_config:
        # Load tokenizer from configuration file
        logger.info(f"Loading tokenizer configuration from {args.tokenizer_config}")
        with open(args.tokenizer_config) as f:
            config_dict = json.load(f)
        config = TokenizerConfig.from_dict(config_dict)
        return Tokenizer(config)

    # Create tokenizer from command-line arguments (Portuguese only now)
    config = TokenizerConfig(
        use_stemming=not args.no_stemming,
        use_stopwords=not args.no_stopwords,
        min_token_length=args.min_token,
        max_token_length=args.max_token,
        keep_numbers=args.keep_numbers,
    )

    logger.info("Created Portuguese tokenizer with:")
    logger.info(f"  - Stemming: {config.use_stemming}")
    logger.info(f"  - Stop words: {config.use_stopwords}")
    logger.info(f"  - Token length: {config.min_token_length}-{config.max_token_length}")
    logger.info(f"  - Keep numbers: {config.keep_numbers}")

    return Tokenizer(config)


def collect_documents(
    input_path: Path, pattern: str, recursive: bool, max_docs: int | None = None
) -> list[Path]:
    """Collect document paths from the input directory.

    This function handles various input formats:
    - Single file
    - Directory of documents
    - File containing list of paths
    """
    documents = []

    if input_path.is_file():
        # Check if it's a list of files or a single document
        if input_path.suffix == ".txt":
            # Could be either a document or a file list
            with open(input_path) as f:
                first_line = f.readline().strip()
                if Path(first_line).exists():
                    # It's a file list
                    f.seek(0)
                    for line in f:
                        doc_path = Path(line.strip())
                        if doc_path.exists():
                            documents.append(doc_path)
                else:
                    # It's a single document
                    documents.append(input_path)
        else:
            documents.append(input_path)

    elif input_path.is_dir():
        # Collect all matching files from directory
        documents = list(input_path.rglob(pattern)) if recursive else list(input_path.glob(pattern))

        # Filter to only files
        documents = [p for p in documents if p.is_file()]

    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    # Apply document limit if specified
    if max_docs and len(documents) > max_docs:
        logger.info(f"Limiting to {max_docs} documents (found {len(documents)})")
        documents = documents[:max_docs]

    return sorted(documents)


def print_statistics(indexer: SPIMIIndexer, elapsed_time: float, timings: dict[str, float]):
    """Print indexing statistics with detailed timing breakdown."""
    print("\n" + "=" * 60)
    print("INDEXING COMPLETE")
    print("=" * 60)
    print(f"Documents indexed: {indexer.doc_count}")
    print(f"Unique terms: {len(indexer.global_term_freq)}")
    print(f"Total tokens processed: {indexer.total_doc_length}")
    avg_length = indexer.total_doc_length / max(1, indexer.doc_count)
    print(f"Average document length: {avg_length:.1f} tokens")
    print(f"Blocks created: {indexer.block_count}")
    print("\n" + "-" * 60)
    print("TIMING BREAKDOWN")
    print("-" * 60)
    tok_time = timings["tokenization_time"]
    tok_pct = tok_time / elapsed_time * 100
    print(f"Tokenization:     {tok_time:>8.2f}s ({tok_pct:>5.1f}%)")

    write_time = timings["block_writing_time"]
    write_pct = write_time / elapsed_time * 100
    print(f"Block writing:    {write_time:>8.2f}s ({write_pct:>5.1f}%)")

    merge_time = timings["merge_time"]
    merge_pct = merge_time / elapsed_time * 100
    print(f"Merging:          {merge_time:>8.2f}s ({merge_pct:>5.1f}%)")

    meta_time = timings["metadata_time"]
    meta_pct = meta_time / elapsed_time * 100
    print(f"Metadata saving:  {meta_time:>8.2f}s ({meta_pct:>5.1f}%)")
    print(f"Total time:       {elapsed_time:>8.2f}s")
    print("\n" + "-" * 60)
    print("THROUGHPUT")
    print("-" * 60)
    print(f"Documents per second: {indexer.doc_count / max(1, elapsed_time):.1f}")
    print(f"Tokens per second:    {indexer.total_doc_length / max(1, elapsed_time):.0f}")
    print("\n" + "=" * 60)
    print(f"Output directory: {indexer.output_dir}")
    print("=" * 60)


def main():
    """Main entry point for the CLI."""
    try:
        # Parse arguments
        args = parse_arguments()

        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Validate memory limit
        if args.memory_limit > 2000:
            logger.error("Memory limit cannot exceed 2000MB due to system constraint")
            sys.exit(1)

        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        # Collect documents
        input_path = Path(args.input_path)
        logger.info(f"Collecting documents from {input_path}")

        documents = collect_documents(input_path, args.file_pattern, args.recursive, args.max_docs)

        if not documents:
            logger.error("No documents found to index")
            sys.exit(1)

        logger.info(f"Found {len(documents)} documents to index")

        # Create tokenizer
        tokenizer = create_tokenizer(args)

        # Save tokenizer configuration for searcher
        config_path = output_path / "tokenizer_config.json"
        with open(config_path, "w") as f:
            json.dump(tokenizer.config.to_dict(), f, indent=2)
        logger.info(f"Saved tokenizer configuration to {config_path}")

        # Create and run indexer
        logger.info("Starting SPIMI indexing...")
        logger.info(f"Memory limit: {args.memory_limit}MB")
        logger.info(f"Block size threshold: {args.block_size}MB")
        logger.info(f"Batch size: {args.batch_size} documents")
        logger.info(f"Minimum term frequency: {args.min_term_freq}")

        indexer = SPIMIIndexer(
            tokenizer=tokenizer,
            output_dir=output_path,
            memory_limit_mb=args.memory_limit,
            min_term_frequency=args.min_term_freq,
            block_size_threshold_mb=args.block_size,
            batch_size=args.batch_size,
            max_docs=args.max_docs,
        )

        # Start indexing with timing
        start_time = time.time()
        indexer.index_collection(documents)
        elapsed_time = time.time() - start_time

        # Get detailed timings from indexer
        timings = indexer.get_timings()

        # Create comprehensive timing statistics
        timing_stats = {
            "configuration": {
                "batch_size": args.batch_size,
                "block_size_mb": args.block_size,
                "memory_limit_mb": args.memory_limit,
                "min_term_frequency": args.min_term_freq,
                "max_docs": args.max_docs,
            },
            "timings": {
                "tokenization_seconds": timings["tokenization_time"],
                "block_writing_seconds": timings["block_writing_time"],
                "merge_seconds": timings["merge_time"],
                "metadata_seconds": timings["metadata_time"],
                "sqlite_insert_seconds": timings["sqlite_insert_time"],
                "arrow_reading_seconds": timings["arrow_reading_time"],
                "term_buffer_flush_seconds": timings["term_buffer_flush_time"],
                "doc_buffer_flush_seconds": timings["doc_buffer_flush_time"],
                "gc_seconds": timings["gc_time"],
                "dict_operations_seconds": timings["dict_operations_time"],
                "total_indexing_seconds": timings["total_indexing_time"],
                "total_elapsed_seconds": elapsed_time,
            },
            "throughput": {
                "documents_per_second": indexer.doc_count / max(1, elapsed_time),
                "tokens_per_second": indexer.total_doc_length / max(1, elapsed_time),
            },
            "statistics": {
                "documents_indexed": indexer.doc_count,
                "unique_terms": len(indexer.global_term_freq),
                "total_tokens": indexer.total_doc_length,
                "blocks_created": indexer.block_count,
                "avg_document_length": indexer.total_doc_length / max(1, indexer.doc_count),
            },
        }

        # Save timing statistics to JSON
        timing_stats_path = output_path / "timing_stats.json"
        with open(timing_stats_path, "w") as f:
            json.dump(timing_stats, f, indent=2)
        logger.info(f"Saved timing statistics to {timing_stats_path}")

        # Print statistics
        print_statistics(indexer, elapsed_time, timings)

        logger.info("Indexing completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("\nIndexing interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
