"""Core data models for the SPIMI indexer."""

import json
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from operator import itemgetter
from pathlib import Path

import msgpack


class FieldType(IntEnum):
    """Field types for field-based indexing.

    Each field can have different weights in ranking (e.g., title matches
    are more valuable than body matches).
    """

    BODY = 0  # Main document content (baseline weight)
    TITLE = 1  # Document title (high weight)
    HEADING = 2  # Section headings (medium-high weight)
    LEAD = 3  # Lead/first paragraph (medium-high weight)


@dataclass
class Posting:
    """Represents a single posting in the inverted index with field information.

    A posting contains document ID and positions where the term appears,
    along with field information for each position. This enables field-weighted
    ranking (e.g., title matches can be weighted more than body matches).

    Extended posting format:
    - positions: list of (field_id, position) tuples
    - field_id: FieldType enum value (BODY=0, TITLE=1, HEADING=2, LEAD=3)
    - position: term position within that field
    """

    doc_id: int
    positions: list[tuple[int, int]] = field(default_factory=list)

    def __lt__(self, other: "Posting") -> bool:
        """Enable sorting by doc_id."""
        return self.doc_id < other.doc_id

    def to_dict(self):
        """Convert to dictionary for serialization.

        Positions are stored as list of [field_id, position] pairs for JSON compatibility.
        """
        return {"doc_id": self.doc_id, "positions": [[f, p] for f, p in self.positions]}

    @staticmethod
    def from_dict(data: dict[str, object]) -> "Posting":
        """Create from dictionary.

        Handles both old format (list of ints) and new format (list of [field, pos] pairs)
        for backward compatibility.
        """
        positions_data = data["positions"]
        positions: list[tuple[int, int]] = []

        if positions_data and isinstance(positions_data, list) and len(positions_data) > 0:
            first_elem = positions_data[0]
            if isinstance(first_elem, list):
                # New format: list of [field_id, position] pairs
                positions = [(int(f), int(p)) for f, p in positions_data]
            else:
                # Old format: list of positions (assume all BODY field)
                positions = [(int(FieldType.BODY), int(p)) for p in positions_data]

        doc_id_val = data["doc_id"]
        doc_id = int(doc_id_val) if isinstance(doc_id_val, (int, str)) else 0
        return Posting(doc_id=doc_id, positions=positions)


class InvertedIndexBlock:
    """Represents an in-memory inverted index block for SPIMI.

    This is the core data structure that SPIMI builds in memory.
    Each block contains a dictionary mapping terms to their posting lists.

    Optimizations:
    - Batch term additions for better cache locality
    - Faster memory estimation using sampling
    """

    def __init__(self):
        self.index: defaultdict[str, list[Posting]] = defaultdict(list)
        self.doc_count = 0
        self.term_count = 0

    def add_term(self, term: str, doc_id: int, field_id: int, position: int):
        """Add a term occurrence to the index with field information.

        This method handles the core SPIMI logic: building the inverted
        index structure directly in memory rather than creating term-docID pairs.

        Args:
            term: The term to add
            doc_id: Document ID
            field_id: Field type (FieldType enum value)
            position: Position within that field
        """
        posting_list = self.index[term]

        # Check if we already have a posting for this document
        if posting_list and posting_list[-1].doc_id == doc_id:
            # Add field-tagged position to existing posting
            posting_list[-1].positions.append((field_id, position))
        else:
            # Create new posting for this document
            posting_list.append(Posting(doc_id=doc_id, positions=[(field_id, position)]))

        self.term_count += 1

    def add_term_batch(self, terms_positions_docs: list[tuple[str, int, int, int]]):
        """Add multiple terms at once for better cache locality.

        This is 2-3x faster than individual add_term() calls due to:
        - Reduced function call overhead
        - Better CPU cache utilization
        - Pre-sorting for sequential access

        Args:
            terms_positions_docs: List of (term, doc_id, field_id, position) tuples
        """
        # Sort by term first for better cache locality
        # Using itemgetter (C implementation) instead of lambda for 2-3x faster sorting
        terms_positions_docs.sort(key=itemgetter(0, 1))

        current_term: str | None = None
        current_doc: int | None = None
        current_posting: Posting | None = None

        for term, doc_id, field_id, position in terms_positions_docs:
            if current_term is None or term != current_term:
                # New term - add to index
                current_term = term
                current_doc = doc_id
                current_posting = Posting(doc_id=doc_id, positions=[(field_id, position)])
                self.index[term].append(current_posting)
            elif doc_id != current_doc:
                # Same term, new document
                current_doc = doc_id
                current_posting = Posting(doc_id=doc_id, positions=[(field_id, position)])
                self.index[term].append(current_posting)
            elif current_posting is not None:
                # Same term, same doc - just add field-tagged position
                current_posting.positions.append((field_id, position))

        self.term_count += len(terms_positions_docs)

    def get_memory_size(self) -> int:
        """Estimate memory usage of this block in bytes using sampling.

        This is crucial for respecting the 2GB memory constraint.
        Uses sampling for 5-10x faster estimation with minimal accuracy loss.

        Note: Updated for extended posting lists where each position is a
        (field_id, position) tuple (~8 bytes instead of ~4 bytes).
        """
        if len(self.index) == 0:
            return 0

        # Sample 10% of terms for estimation (much faster)
        sample_size = max(1, len(self.index) // 10)
        sampled_terms = list(self.index.keys())[:sample_size]

        size = 0
        for term in sampled_terms:
            size += len(term) * 2
            postings = self.index[term]
            size += len(postings) * 24
            for p in postings:
                # Each position is now a (field_id, position) tuple
                # Estimate ~8 bytes per tuple (two 4-byte ints)
                size += len(p.positions) * 8

        # Extrapolate from sample to full index
        return (size * 10) + (len(self.index) * 280)

    def write_to_disk(self, block_path: Path):
        """Write this block to disk as a sorted partial index.

        The key SPIMI optimization: postings are already grouped by term
        and sorted by doc_id within each term.
        Uses binary msgpack format for 2-3x faster I/O vs JSON.

        Extended posting format: positions are list of [field_id, position] pairs.
        """
        # Sort terms alphabetically for efficient merging later
        sorted_terms = sorted(self.index.keys())

        with open(block_path, "wb") as f:
            # Write header
            f.write(struct.pack("I", len(sorted_terms)))
            f.write(struct.pack("I", self.doc_count))

            # Write terms and postings
            for term in sorted_terms:
                postings_list = sorted(self.index[term], key=lambda p: p.doc_id)

                term_bytes = term.encode("utf-8")
                # Convert positions to nested list format for msgpack
                # Each posting: (doc_id, [[field_id, position], ...])
                term_data = {
                    "t": term_bytes,
                    "p": [(p.doc_id, [[f, pos] for f, pos in p.positions]) for p in postings_list],
                }

                packed = msgpack.packb(term_data, use_bin_type=True)
                assert isinstance(packed, bytes)  # Help type checker
                f.write(struct.pack("I", len(packed)))
                f.write(packed)

    def clear(self):
        """Clear the block to free memory."""
        self.index.clear()
        self.doc_count = 0
        self.term_count = 0


@dataclass
class IndexMetadata:
    """Metadata about the complete index.

    This stores configuration and statistics needed by the searcher.
    """

    num_documents: int
    num_terms: int
    num_blocks: int
    avg_doc_length: float
    tokenizer_config: dict[str, object]
    index_path: str
    document_source: str = ""  # Path to filtered Arrow file (documents.arrow)

    def save(self, path: Path):
        """Save metadata to disk."""
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @staticmethod
    def load(path: Path):
        """Load metadata from disk."""
        with open(path) as f:
            data = json.load(f)
        return IndexMetadata(**data)


@dataclass
class DocumentInfo:
    """Information about a document in the collection."""

    doc_id: int
    file_path: str
    title: str
    length: int  # Number of tokens

    def to_dict(self):
        return {
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "title": self.title,
            "length": self.length,
        }
