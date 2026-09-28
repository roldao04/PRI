"""SPIMI (Single-Pass In-Memory Indexing) implementation.

This module implements the core indexing algorithm that respects
the 2GB memory constraint while efficiently building an inverted index
for large document collections.

Optimizations:
- Binary msgpack format for intermediate blocks (2-3x faster I/O)
- Final index kept in JSONL for compatibility with existing search code
- Batch processing with larger buffers
- Better memory management and reduced object overhead
- Explicit garbage collection to free memory promptly
"""

import gc
import heapq
import json
import logging
import sqlite3
import struct
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import BinaryIO

import msgpack

from indexa.indexer.document_reader import ArrowDocumentReader
from indexa.indexer.field_extractor import FieldExtractor
from indexa.indexer.models import DocumentInfo, IndexMetadata, InvertedIndexBlock, Posting
from indexa.indexer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class SPIMIIndexer:
    """Single-Pass In-Memory Indexer implementation.

    This class implements the SPIMI algorithm which:
    1. Reads documents sequentially
    2. Builds inverted index blocks in memory
    3. Writes blocks to disk when memory limit is approached
    4. Merges all blocks into final index

    The key advantage over BSBI is avoiding the expensive global sort
    by building sorted structures directly in memory.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        output_dir: Path,
        memory_limit_mb: int = 1800,  # Leave buffer below 2GB
        min_term_frequency: int = 1,
        block_size_threshold_mb: int = 120,  # Reduced for extended postings (field-based indexing)
        max_docs: int | None = None,
        batch_size: int = 1000,  # Documents per batch for processing
    ):
        """Initialize the SPIMI indexer.

        Args:
            tokenizer: Tokenizer for processing documents
            output_dir: Directory to store index files
            memory_limit_mb: Memory limit in MB (default 1800 to stay under 2GB)
            min_term_frequency: Minimum term frequency for index (applied during merge)
            block_size_threshold_mb: Write block when it reaches this size (default 120MB,
                                    reduced for field-based indexing which uses
                                    ~2x memory per position)
            max_docs: Maximum number of documents to index (for testing)
            batch_size: Number of documents to process in each batch (default 1000)
        """
        self.tokenizer = tokenizer
        self.field_extractor = FieldExtractor()
        self.output_dir = Path(output_dir)
        self.memory_limit_mb = memory_limit_mb
        self.min_term_frequency = min_term_frequency
        self.block_size_threshold_mb = block_size_threshold_mb
        self.max_docs = max_docs
        self.batch_size = batch_size

        # Create output directory structure
        self.blocks_dir = self.output_dir / "blocks"
        self.final_dir = self.output_dir / "final"
        self.blocks_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)

        # Initialize indexing state
        self.current_block = InvertedIndexBlock()
        self.block_count = 0
        self.doc_count = 0
        self.total_doc_length = 0
        # Stream doc_info to disk instead of keeping in memory
        # File stays open for entire indexing process, closed in _save_metadata()
        self.doc_info_path = self.output_dir / "documents.jsonl"
        self.doc_info_file = open(self.doc_info_path, "w", encoding="utf-8")  # noqa: SIM115
        self.global_term_freq: defaultdict[str, int] = defaultdict(int)

        # Batch processing buffer - now stores (term, doc_id, field_id, position)
        self.term_buffer: list[tuple[str, int, int, int]] = []
        # Reduced from 50K due to extended postings (4 elements per tuple vs 3)
        self.TERM_BUFFER_SIZE = 30000

        # Track Arrow file source path for metadata
        self.arrow_source_path: Path | None = None

        # SQLite forward index for fast document retrieval
        self.db_path = self.output_dir / "documents.db"
        # Remove existing database if it exists (fresh indexing)
        if self.db_path.exists():
            self.db_path.unlink()
        self.db_connection = sqlite3.connect(str(self.db_path))
        self.db_cursor = self.db_connection.cursor()

        # Create documents table with doc_id as primary key
        self.db_cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL
            )
        """
        )
        self.db_connection.commit()

        # Document buffer for batch SQLite inserts
        self.doc_buffer: list[tuple[int, str, str]] = []
        self.DOC_BUFFER_SIZE = 1000  # Batch insert every 1000 docs

        # Timing tracking
        self.timings = {
            "tokenization_time": 0.0,
            "block_writing_time": 0.0,
            "merge_time": 0.0,
            "metadata_time": 0.0,
            "sqlite_insert_time": 0.0,
            "arrow_reading_time": 0.0,
            "term_buffer_flush_time": 0.0,
            "doc_buffer_flush_time": 0.0,
            "gc_time": 0.0,
            "dict_operations_time": 0.0,
            "total_indexing_time": 0.0,
        }

        logger.info(f"Initialized SPIMI indexer with {memory_limit_mb}MB limit")
        logger.info(f"Created SQLite forward index database: {self.db_path}")

    def index_collection(self, document_paths: list[Path]):
        """Index a collection of documents using SPIMI.

        This is the main entry point that orchestrates the entire
        indexing process following the SPIMI algorithm.
        """
        start_time = time.time()
        logger.info(f"Starting indexing of {len(document_paths)} documents")

        # Phase 1: Build partial index blocks
        for doc_path in document_paths:
            if self.max_docs is not None and self.doc_count >= self.max_docs:
                logger.info(f"Reached max_docs limit of {self.max_docs}, stopping indexing")
                break

            if doc_path.suffix.lower() in {".arrow", ".feather"}:
                self._index_arrow_file(doc_path)
            else:
                self._index_document(doc_path)

            # SPIMI decision point: write block when threshold reached
            block_size_mb = self.current_block.get_memory_size() / (1024 * 1024)
            if block_size_mb >= self.block_size_threshold_mb:
                self._write_current_block()

        if self.term_buffer:
            self._flush_term_buffer()

        if self.current_block.term_count > 0:
            self._write_current_block()

        # Phase 2: Merge blocks
        logger.info(f"Created {self.block_count} blocks, starting merge phase")
        self._merge_blocks()

        # Flush any remaining documents to SQLite
        if self.doc_buffer:
            logger.info(f"Flushing remaining {len(self.doc_buffer)} documents to SQLite")
            self._flush_documents_to_sqlite()

        # Phase 3: Save metadata (also closes doc_info file and SQLite database)
        self._save_metadata()

        self.timings["total_indexing_time"] = time.time() - start_time

        logger.info(
            f"Indexing complete: {self.doc_count} documents, "
            f"{len(self.global_term_freq)} unique terms"
        )

    def __del__(self):
        """Ensure doc_info file and database are closed when indexer is destroyed."""
        if hasattr(self, "doc_info_file") and not self.doc_info_file.closed:
            self.doc_info_file.close()
        if hasattr(self, "db_connection"):
            with suppress(Exception):
                self.db_connection.close()

    def _flush_term_buffer(self):
        """Flush the term buffer to the current block."""
        if self.term_buffer:
            flush_start = time.time()
            self.current_block.add_term_batch(self.term_buffer)
            self.term_buffer = []
            self.timings["term_buffer_flush_time"] += time.time() - flush_start

    def _flush_documents_to_sqlite(self):
        """Flush document buffer to SQLite database with batch insert."""
        if not self.doc_buffer:
            return

        sqlite_start = time.time()

        try:
            # Use executemany for batch insert (much faster than individual inserts)
            self.db_cursor.executemany(
                "INSERT OR REPLACE INTO documents (doc_id, title, content) VALUES (?, ?, ?)",
                self.doc_buffer,
            )
            self.db_connection.commit()

            logger.debug(f"Flushed {len(self.doc_buffer)} documents to SQLite")
            self.doc_buffer.clear()

        except Exception as e:
            logger.error(f"Failed to flush documents to SQLite: {e}")

        elapsed = time.time() - sqlite_start
        self.timings["sqlite_insert_time"] += elapsed
        self.timings["doc_buffer_flush_time"] += elapsed

    def _save_document_info(self, doc_info: DocumentInfo):
        """Save document info incrementally to disk.

        This streams document metadata to a JSONL file instead of
        keeping everything in memory, significantly reducing memory usage.
        """
        self.doc_info_file.write(json.dumps(doc_info.to_dict()) + "\n")

    def _index_document(self, doc_path: Path):
        """Index a single document into the current block.

        This method implements the core SPIMI logic: building
        the inverted index structure directly in memory.

        Now uses field-based indexing to extract and weight different
        document fields (title, lead, headings, body).
        """
        try:
            # Read document content
            with open(doc_path, encoding="utf-8") as f:
                content = f.read()

            doc_id = self.doc_count
            self.doc_count += 1

            title = doc_path.stem

            # Extract structured fields
            fields = self.field_extractor.extract_fields(title, content)

            # Tokenize with field information
            tokenization_start = time.time()
            tokens_with_field_pos = list(self.tokenizer.tokenize_fields(fields))
            self.timings["tokenization_time"] += time.time() - tokenization_start

            doc_length = len(tokens_with_field_pos)

            doc_info = DocumentInfo(
                doc_id=doc_id,
                file_path=str(doc_path),
                title=title,
                length=doc_length,
            )
            self._save_document_info(doc_info)
            self.total_doc_length += doc_length

            # Buffer for SQLite forward index
            self.doc_buffer.append((doc_id, title, content))

            if len(self.doc_buffer) >= self.DOC_BUFFER_SIZE:
                self._flush_documents_to_sqlite()

            # Build inverted index directly (SPIMI approach) with field information
            for term, field_id, position in tokens_with_field_pos:
                self.current_block.add_term(term, doc_id, field_id, position)
                self.global_term_freq[term] += 1

            self.current_block.doc_count += 1

            if self.doc_count % 100 == 0:
                logger.info(f"Indexed {self.doc_count} documents")

        except Exception as e:
            logger.error(f"Error indexing {doc_path}: {e}")

    def _index_arrow_file(self, arrow_path: Path):
        """Index all documents from an Arrow file.

        This method streams through the Arrow file and processes documents
        in batches for better performance while staying within memory constraints.
        Optimized with larger batches and less frequent memory checks.

        Uses single-pass streaming to filter redirects during iteration.
        """
        try:
            reader = ArrowDocumentReader(filter_redirects=True)

            # Store Arrow source path for metadata
            self.arrow_source_path = arrow_path

            logger.info("Indexing documents from Arrow file (streaming mode)")
            logger.info(f"Using batch size: {self.batch_size}")

            batch = []
            docs_processed = 0

            # Process documents with single-pass streaming
            for doc_data in reader.read_documents(arrow_path):
                # Time document reading from Arrow
                arrow_read_start = time.time()

                # Apply max_docs limit during iteration
                if self.max_docs is not None and self.doc_count >= self.max_docs:
                    logger.info(f"Reached max_docs limit of {self.max_docs}, stopping indexing")
                    break

                content = doc_data["content"]
                title = doc_data["title"]
                global_row_idx = doc_data["metadata"]["global_row_idx"]

                if not content:
                    self.timings["arrow_reading_time"] += time.time() - arrow_read_start
                    continue

                batch.append((content, title, global_row_idx))
                docs_processed += 1

                self.timings["arrow_reading_time"] += time.time() - arrow_read_start

                if len(batch) >= self.batch_size:
                    self._process_document_batch(batch, arrow_path)
                    batch = []

                    # Check memory more frequently with extended postings
                    # (every 5K docs instead of 10K)
                    if self.doc_count % 5000 == 0:
                        block_size_mb = self.current_block.get_memory_size() / (1024 * 1024)
                        if block_size_mb >= self.block_size_threshold_mb:
                            self._flush_term_buffer()
                            self._write_current_block()
                        else:
                            # Even if not writing block, trigger GC periodically
                            gc_start = time.time()
                            gc.collect()
                            self.timings["gc_time"] += time.time() - gc_start

            if batch:
                self._process_document_batch(batch, arrow_path)

            if self.term_buffer:
                self._flush_term_buffer()

        except Exception as e:
            logger.error(f"Error indexing Arrow file {arrow_path}: {e}")

    def _process_document_batch(self, batch: list[tuple[str, str, int]], arrow_path: Path):
        """Process a batch of documents for indexing with buffered term additions.

        Uses batch field extraction and tokenization to process multiple documents,
        reducing overhead by 20-30%.

        Now uses field-based indexing for improved ranking.

        Args:
            batch: List of (content, title, global_row_idx) tuples
            arrow_path: Path to the Arrow file being processed
        """
        # Extract fields for all documents in batch
        fields_list = []
        for content, title, _ in batch:
            fields = self.field_extractor.extract_fields(title, content)
            fields_list.append(fields)

        # Batch tokenize all documents at once with field information
        tokenization_start = time.time()
        all_tokens_with_field_pos = self.tokenizer.tokenize_fields_batch(fields_list)
        self.timings["tokenization_time"] += time.time() - tokenization_start

        # Process each document's tokens
        for (content, title, global_row_idx), tokens_with_field_pos in zip(
            batch, all_tokens_with_field_pos
        ):
            # Use the original Arrow file row index as doc_id
            doc_id = global_row_idx
            self.doc_count += 1  # Still increment for counting purposes

            doc_length = len(tokens_with_field_pos)

            doc_info = DocumentInfo(
                doc_id=doc_id, file_path=str(arrow_path), title=title, length=doc_length
            )
            self._save_document_info(doc_info)
            self.total_doc_length += doc_length

            # Buffer for SQLite forward index
            self.doc_buffer.append((doc_id, title, content))

            # Time term buffering and dictionary operations
            dict_start = time.time()
            for term, field_id, position in tokens_with_field_pos:
                self.term_buffer.append((term, doc_id, field_id, position))
                self.global_term_freq[term] += 1
            self.timings["dict_operations_time"] += time.time() - dict_start

            self.current_block.doc_count += 1

        # Clear large intermediate lists to free memory
        fields_list.clear()
        all_tokens_with_field_pos.clear()

        if len(self.term_buffer) >= self.TERM_BUFFER_SIZE:
            self._flush_term_buffer()

        # Flush SQLite buffer if needed
        if len(self.doc_buffer) >= self.DOC_BUFFER_SIZE:
            self._flush_documents_to_sqlite()

        if self.doc_count % 5000 == 0:
            logger.info(f"Indexed {self.doc_count} documents from Arrow file")

    def _write_current_block(self):
        """Write the current in-memory block to disk.

        This is the "freeze" and "write to disk" step in SPIMI.
        The block is written as a sorted partial index using msgpack.
        """
        if self.current_block.term_count == 0:
            return

        write_start = time.time()
        block_path = self.blocks_dir / f"block_{self.block_count:04d}.msgpack"

        logger.info(
            f"Writing block {self.block_count} with "
            f"{len(self.current_block.index)} terms to {block_path}"
        )

        self.current_block.write_to_disk(block_path)

        # Clear block to free memory
        self.current_block.clear()
        self.current_block = InvertedIndexBlock()
        self.block_count += 1

        # Explicitly trigger garbage collection to free memory promptly
        gc_start = time.time()
        gc.collect()
        self.timings["gc_time"] += time.time() - gc_start

        self.timings["block_writing_time"] += time.time() - write_start

    def _merge_blocks(self):
        """Merge all partial index blocks into final index.

        This implements the multi-way merge step of SPIMI.
        The beauty is that each block is already sorted by term,
        so we can efficiently merge them without loading everything
        into memory at once.

        Uses streaming to write results incrementally instead of building
        a large in-memory dictionary, preventing memory spikes.
        Now uses binary format for faster I/O.
        Uses context managers for proper file handle cleanup.
        """
        merge_start = time.time()
        block_paths = sorted(self.blocks_dir.glob("block_*.msgpack"))

        if len(block_paths) == 0:
            logger.warning("No blocks to merge")
            return

        logger.info(f"Merging {len(block_paths)} blocks")

        # Open all blocks using context managers for proper cleanup
        # ExitStack allows us to manage multiple context managers dynamically
        with ExitStack() as stack:
            block_readers: list[BlockReader] = []
            for block_path in block_paths:
                reader = stack.enter_context(BlockReader(block_path))
                block_readers.append(reader)

            # Create the final merged index using k-way merge
            # Write directly to disk instead of building in-memory dict
            merger = KWayMerger(block_readers, self.min_term_frequency)

            # Stream merge and write incrementally to prevent memory spike
            self._save_final_index_streaming(merger)

        # All BlockReaders are now properly closed by ExitStack

        self.timings["merge_time"] = time.time() - merge_start

        # Clean up block files
        for block_path in block_paths:
            block_path.unlink()

        # Trigger garbage collection after merge and cleanup
        gc_start = time.time()
        gc.collect()
        self.timings["gc_time"] += time.time() - gc_start

    def _save_final_index_streaming(self, merger: "KWayMerger"):
        """Save the final merged index to disk using streaming with JSONL format.

        Writes terms one at a time as JSONL to avoid building large in-memory dictionary.
        This prevents the memory spike during merge.
        Uses JSONL format for compatibility with existing search code.
        """
        index_path = self.final_dir / "index.jsonl"
        term_count = 0

        with open(index_path, "w", encoding="utf-8") as f:
            for term, merged_postings in merger.merge():
                term_freq = self.global_term_freq[term]
                if term_freq >= self.min_term_frequency:
                    term_data = {
                        "term": term,
                        "postings": [posting.to_dict() for posting in merged_postings],
                    }
                    f.write(json.dumps(term_data, ensure_ascii=False) + "\n")
                    term_count += 1

                    if term_count % 50000 == 0:
                        logger.info(f"Merged and saved {term_count} terms")

        logger.info(f"Saved final index with {term_count} terms to {index_path}")

    def _save_metadata(self):
        """Save index metadata and document information."""
        metadata_start = time.time()

        # Close the streaming doc_info file
        self.doc_info_file.close()

        # Flush and close SQLite database
        if self.doc_buffer:
            self._flush_documents_to_sqlite()

        self.db_connection.close()
        logger.info(f"Closed SQLite database: {self.db_path}")

        # Determine the document source path
        # Use original Arrow file since doc_ids are original row indices
        doc_source_path = str(self.arrow_source_path) if self.arrow_source_path else ""

        filtered_terms = [
            t for t in self.global_term_freq if self.global_term_freq[t] >= self.min_term_frequency
        ]
        metadata = IndexMetadata(
            num_documents=self.doc_count,
            num_terms=len(filtered_terms),
            num_blocks=self.block_count,
            avg_doc_length=self.total_doc_length / max(1, self.doc_count),
            tokenizer_config=self.tokenizer.config.to_dict(),
            index_path=str(self.final_dir),
            document_source=doc_source_path,
        )

        metadata_path = self.output_dir / "metadata.json"
        metadata.save(metadata_path)

        # Convert JSONL to JSON for compatibility with existing code
        logger.info("Converting document info from JSONL to JSON...")
        doc_info_dict: dict[str, object] = {}
        with open(self.doc_info_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    doc_data = json.loads(line)
                    doc_id = str(doc_data["doc_id"])
                    # Update file_path to point to original Arrow file
                    doc_data["file_path"] = doc_source_path
                    doc_info_dict[doc_id] = doc_data

        doc_info_json_path = self.output_dir / "documents.json"
        with open(doc_info_json_path, "w", encoding="utf-8") as f:
            json.dump(doc_info_dict, f, indent=2)

        # Remove the JSONL file to save space
        self.doc_info_path.unlink()

        self.timings["metadata_time"] = time.time() - metadata_start

        logger.info("Saved metadata and document information")

    def get_timings(self) -> dict[str, float]:
        """Return timing statistics for all indexing phases.

        Returns:
            Dictionary with timing information for each phase in seconds
        """
        return self.timings.copy()


class BlockReader:
    """Reader for streaming terms from a block file using binary format.

    This class enables memory-efficient merging by reading
    blocks one term at a time rather than loading entire blocks.
    Uses msgpack binary format for 2-3x faster I/O.

    Can be used as a context manager to ensure proper file handle cleanup.
    """

    def __init__(self, block_path: Path):
        self.block_path = block_path
        self.file_handle: BinaryIO = open(block_path, "rb")  # noqa: SIM115

        # Read header with metadata
        self.num_terms = struct.unpack("I", self.file_handle.read(4))[0]
        self.doc_count = struct.unpack("I", self.file_handle.read(4))[0]

        # Cache for peek operation
        self._next_term_data: dict[str, object] | None = None
        self._load_next()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and ensure file is closed."""
        self.close()
        return False

    def _load_next(self):
        """Load the next term efficiently from binary format."""
        try:
            length_bytes = self.file_handle.read(4)
            if len(length_bytes) < 4:
                self._next_term_data = None
                return

            length = struct.unpack("I", length_bytes)[0]
            packed_data = self.file_handle.read(length)

            if len(packed_data) < length:
                self._next_term_data = None
                return

            self._next_term_data = msgpack.unpackb(packed_data, raw=False)
        except Exception:
            self._next_term_data = None

    def has_next(self) -> bool:
        """Check if there are more terms."""
        return self._next_term_data is not None

    def peek_term(self) -> str | None:
        """Look at the next term without consuming it."""
        if self._next_term_data:
            term_bytes = self._next_term_data["t"]
            if isinstance(term_bytes, bytes):
                return term_bytes.decode("utf-8")
            return str(term_bytes)
        return None

    def get_next(self) -> tuple[str | None, list[Posting]]:
        """Get the next term and its postings."""
        if not self._next_term_data:
            return None, []

        data = self._next_term_data
        term_bytes = data["t"]
        term = term_bytes.decode("utf-8") if isinstance(term_bytes, bytes) else str(term_bytes)
        postings_data = data["p"]
        postings = [
            Posting(doc_id=doc_id, positions=list(positions))
            for doc_id, positions in postings_data  # type: ignore[attr-defined]
        ]

        # Load next term for future peek
        self._load_next()

        return term, postings

    def close(self):
        """Close the file handle."""
        if self.file_handle:
            self.file_handle.close()

    def __del__(self):
        """Ensure file handle is closed."""
        self.close()


class KWayMerger:
    """K-way merger for combining multiple sorted block files.

    This implements the efficient multi-way merge algorithm
    that combines sorted blocks without loading them all into memory.
    It's like merging multiple sorted lists simultaneously.
    """

    def __init__(self, block_readers: list[BlockReader], min_term_freq: int = 1):
        self.block_readers = block_readers
        self.min_term_freq = min_term_freq

    def merge(self) -> Iterator[tuple[str, list[Posting]]]:
        """Merge all blocks, yielding merged term-postings pairs.

        This uses a min-heap to efficiently find the next term
        across all blocks, similar to the merge step in merge sort.
        """
        # Initialize heap with first term from each block
        heap: list[tuple[str, int]] = []
        for i, reader in enumerate(self.block_readers):
            if reader.has_next():
                term = reader.peek_term()
                if term is not None:
                    heapq.heappush(heap, (term, i))

        while heap:
            # Get the minimum term across all blocks
            current_term, _ = heap[0]
            merged_postings: list[Posting] = []

            # Collect postings for this term from all blocks that have it
            blocks_to_update: list[tuple[str, int]] = []
            while heap and heap[0][0] == current_term:
                _, block_idx = heapq.heappop(heap)
                reader = self.block_readers[block_idx]

                _, postings = reader.get_next()
                merged_postings.extend(postings)

                # Remember to add next term from this block
                if reader.has_next():
                    next_term = reader.peek_term()
                    if next_term is not None:
                        blocks_to_update.append((next_term, block_idx))

            # Add next terms back to heap
            for term, idx in blocks_to_update:
                heapq.heappush(heap, (term, idx))

            # Sort merged postings by doc_id
            merged_postings.sort()

            yield current_term, merged_postings
