"""Index loader for reading and managing the inverted index.

This module provides efficient loading and querying of the inverted index
created by the SPIMI indexer. It supports loading the full index into memory
for fast query processing.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from indexa.indexer.models import DocumentInfo, IndexMetadata, Posting

logger = logging.getLogger(__name__)


class InvertedIndex:
    """In-memory representation of the inverted index.

    This class stores the complete inverted index in memory for fast querying.
    Since the searcher has no memory constraint, we can load everything for
    maximum performance.
    """

    def __init__(self):
        """Initialize empty index."""
        # Main index: term -> list of postings
        self.index: dict[str, list[Posting]] = {}

        # Document frequency: term -> number of documents containing term
        self.doc_frequency: dict[str, int] = {}

        # Metadata
        self.metadata: IndexMetadata | None = None

        # Document information
        self.documents: dict[int, DocumentInfo] = {}

    def get_postings(self, term: str) -> list[Posting]:
        """Get postings list for a term.

        Args:
            term: The term to look up

        Returns:
            List of postings for the term (empty if term not in index)
        """
        return self.index.get(term, [])

    def get_doc_frequency(self, term: str) -> int:
        """Get document frequency for a term.

        Args:
            term: The term to look up

        Returns:
            Number of documents containing the term
        """
        return self.doc_frequency.get(term, 0)

    def get_document_info(self, doc_id: int) -> DocumentInfo | None:
        """Get information about a document.

        Args:
            doc_id: The document ID

        Returns:
            DocumentInfo object or None if not found
        """
        return self.documents.get(doc_id)

    def get_term_frequencies_for_doc(self, doc_id: int, terms: list[str]) -> dict[str, int]:
        """Get term frequencies for specific terms in a document.

        Args:
            doc_id: The document ID
            terms: List of terms to look up

        Returns:
            Dictionary mapping terms to their frequency in the document
        """
        term_freqs: dict[str, int] = {}

        for term in terms:
            postings = self.get_postings(term)
            for posting in postings:
                if posting.doc_id == doc_id:
                    # Frequency is the number of positions
                    term_freqs[term] = len(posting.positions)
                    break

        return term_freqs

    def get_document_vector(self, doc_id: int) -> dict[str, int]:
        """Get the full term frequency vector for a document.

        This is useful for relevance feedback where we need to find similar documents.

        Args:
            doc_id: The document ID

        Returns:
            Dictionary mapping all terms in the document to their frequencies
        """
        doc_vector: defaultdict[str, int] = defaultdict(int)

        # Iterate through entire index to find all terms in this document
        for term, postings in self.index.items():
            for posting in postings:
                if posting.doc_id == doc_id:
                    doc_vector[term] = len(posting.positions)
                    break

        return dict(doc_vector)

    def get_all_terms(self) -> list[str]:
        """Get all terms in the index.

        Returns:
            List of all terms
        """
        return list(self.index.keys())

    def get_num_documents(self) -> int:
        """Get total number of documents in the index."""
        return len(self.documents)

    def get_avg_doc_length(self) -> float:
        """Get average document length."""
        if self.metadata:
            return self.metadata.avg_doc_length
        return 0.0


class IndexLoader:
    """Loader for reading the inverted index from disk.

    This class handles loading the JSONL format index created by SPIMI
    and building an efficient in-memory representation.
    """

    def __init__(self, index_dir: Path):
        """Initialize index loader.

        Args:
            index_dir: Directory containing the index files
        """
        self.index_dir = Path(index_dir)

    def load(self) -> InvertedIndex:
        """Load the complete index into memory.

        Returns:
            InvertedIndex object with all data loaded

        Raises:
            FileNotFoundError: If index files are missing
            ValueError: If index format is invalid
        """
        logger.info(f"Loading index from {self.index_dir}")

        index = InvertedIndex()

        # Load metadata
        metadata_path = self.index_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        index.metadata = IndexMetadata.load(metadata_path)
        logger.info(
            f"Loaded metadata: {index.metadata.num_documents} documents, "
            f"{index.metadata.num_terms} terms"
        )

        # Load document information
        documents_path = self.index_dir / "documents.json"
        if not documents_path.exists():
            raise FileNotFoundError(f"Documents file not found: {documents_path}")

        with open(documents_path, encoding="utf-8") as f:
            doc_data = json.load(f)

        for doc_id_str, doc_info_dict in doc_data.items():
            doc_id = int(doc_id_str)
            index.documents[doc_id] = DocumentInfo(
                doc_id=doc_info_dict["doc_id"],
                file_path=doc_info_dict["file_path"],
                title=doc_info_dict["title"],
                length=doc_info_dict["length"],
            )

        logger.info(f"Loaded {len(index.documents)} document records")

        # Load inverted index from JSONL
        index_path = self.index_dir / "final" / "index.jsonl"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")

        term_count = 0
        with open(index_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                term_data = json.loads(line)
                term = term_data["term"]
                postings_data = term_data["postings"]

                # Convert postings data to Posting objects
                # Use from_dict to handle both old and new formats (with field information)
                postings = [Posting.from_dict(p) for p in postings_data]

                index.index[term] = postings
                index.doc_frequency[term] = len(postings)

                term_count += 1
                if term_count % 100000 == 0:
                    logger.info(f"Loaded {term_count} terms...")

        logger.info(f"Loaded complete index with {term_count} terms")

        return index

    def load_lazy(self) -> "LazyInvertedIndex":
        """Load index with lazy loading (for very large indices).

        Returns:
            LazyInvertedIndex that loads terms on-demand
        """
        return LazyInvertedIndex(self.index_dir)


class LazyInvertedIndex(InvertedIndex):
    """Lazy-loading version of the inverted index.

    This version doesn't load the entire index into memory upfront.
    Instead, it loads terms on-demand as they're queried.
    Useful for extremely large indices where memory is a concern.
    """

    def __init__(self, index_dir: Path):
        """Initialize lazy index.

        Args:
            index_dir: Directory containing the index files
        """
        super().__init__()
        self.index_dir = Path(index_dir)

        logger.info(f"Initializing lazy index from {index_dir}")

        # Load metadata and documents immediately (they're small)
        logger.info("Loading metadata...")
        metadata_path = self.index_dir / "metadata.json"
        self.metadata = IndexMetadata.load(metadata_path)
        logger.info(
            f"Loaded metadata: {self.metadata.num_documents} documents, "
            f"{self.metadata.num_terms} terms"
        )

        logger.info("Loading document information...")
        documents_path = self.index_dir / "documents.json"
        with open(documents_path, encoding="utf-8") as f:
            doc_data = json.load(f)

        for doc_id_str, doc_info_dict in doc_data.items():
            doc_id = int(doc_id_str)
            self.documents[doc_id] = DocumentInfo(
                doc_id=doc_info_dict["doc_id"],
                file_path=doc_info_dict["file_path"],
                title=doc_info_dict["title"],
                length=doc_info_dict["length"],
            )

        logger.info(f"Loaded {len(self.documents)} document records")

        # Build term offset index for fast seeking
        logger.info("Building term offset index (this may take a few minutes for large indices)...")
        self._build_term_index()

        logger.info("Initialized lazy index loader successfully")

    def _build_term_index(self):
        """Build an index of term positions in the JSONL file for fast seeking."""
        self.term_offsets: dict[str, int] = {}

        index_path = self.index_dir / "final" / "index.jsonl"

        # Get file size for progress reporting
        file_size = index_path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        logger.info(f"Scanning index file: {file_size_mb:.1f} MB ({index_path.name})")

        with open(index_path, "rb") as f:
            offset = 0
            term_count = 0
            last_log_at = 0

            for line in f:
                if not line.strip():
                    offset = f.tell()
                    continue

                # Parse just enough to get the term
                line_str = line.decode("utf-8")
                term_data = json.loads(line_str)
                term = term_data["term"]

                self.term_offsets[term] = offset
                self.doc_frequency[term] = len(term_data["postings"])

                offset = f.tell()
                term_count += 1

                # Log progress every 50k terms
                if term_count - last_log_at >= 50000:
                    progress_pct = (offset / file_size) * 100
                    logger.info(
                        f"Progress: {term_count:,} terms indexed "
                        f"({progress_pct:.1f}% of file scanned)"
                    )
                    last_log_at = term_count

        logger.info(f"Built term offset index with {len(self.term_offsets):,} terms")

    def get_postings(self, term: str) -> list[Posting]:
        """Get postings for a term (loads from disk if not cached).

        Args:
            term: The term to look up

        Returns:
            List of postings for the term
        """
        # Check if already loaded
        if term in self.index:
            logger.debug(f"Cache HIT for term '{term}' ({len(self.index[term])} postings)")
            return self.index[term]

        # Load from disk
        if term not in self.term_offsets:
            logger.debug(f"Term '{term}' not found in index")
            return []

        logger.debug(f"Cache MISS for term '{term}' - loading from disk...")
        index_path = self.index_dir / "final" / "index.jsonl"
        offset = self.term_offsets[term]

        with open(index_path, encoding="utf-8") as f:
            f.seek(offset)
            line = f.readline()
            term_data = json.loads(line)

            # Use from_dict to handle both old and new formats (with field information)
            postings = [Posting.from_dict(p) for p in term_data["postings"]]

            # Cache for future use
            self.index[term] = postings
            logger.debug(f"Loaded and cached {len(postings)} postings for term '{term}'")

            return postings
