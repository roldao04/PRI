"""BM25 ranking algorithm implementation.

BM25 (Best Match 25) is a probabilistic ranking function used by search engines
to estimate the relevance of documents to a given search query.

Formula:
BM25(D, Q) = Σ IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

Where:
- D: document
- Q: query
- qi: query term
- f(qi, D): frequency of term qi in document D
- |D|: length of document D (in tokens)
- avgdl: average document length in the collection
- k1: term frequency saturation parameter (typical: 1.2-2.0)
- b: length normalization parameter (typical: 0.75)
- IDF(qi): inverse document frequency of term qi

BM25F Extension:
BM25F extends BM25 by applying field-specific weights to term occurrences.
This allows title matches to be weighted more heavily than body matches.
"""

import math
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from indexa.indexer.models import Posting


class BM25Scorer:
    """BM25 ranking algorithm implementation.

    This class provides scoring functionality using the BM25 algorithm,
    which balances term frequency, document length, and term rarity.
    """

    def __init__(
        self,
        num_documents: int,
        avg_doc_length: float,
        k1: float = 1.2,
        b: float = 0.75,
    ):
        """Initialize BM25 scorer.

        Args:
            num_documents: Total number of documents in the collection
            avg_doc_length: Average document length (in tokens)
            k1: Term frequency saturation parameter (default: 1.2)
            b: Length normalization parameter (default: 0.75)
        """
        self.num_documents = num_documents
        self.avg_doc_length = avg_doc_length
        self.k1 = k1
        self.b = b

        # Cache IDF values for efficiency
        self.idf_cache: dict[str, float] = {}

    def compute_idf(self, term: str, doc_frequency: int) -> float:
        """Compute IDF (Inverse Document Frequency) for a term.

        IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        where N is total documents, df is document frequency.

        Args:
            term: The term to compute IDF for
            doc_frequency: Number of documents containing the term

        Returns:
            IDF score for the term
        """
        if term in self.idf_cache:
            return self.idf_cache[term]

        # Standard BM25 IDF formula with smoothing
        idf = math.log((self.num_documents - doc_frequency + 0.5) / (doc_frequency + 0.5) + 1.0)

        self.idf_cache[term] = idf
        return idf

    def score_document(
        self,
        query_terms: list[str],
        term_frequencies: dict[str, int],
        doc_length: int | float,
        term_doc_frequencies: dict[str, int],
    ) -> float:
        """Calculate BM25 score for a document given a query.

        Args:
            query_terms: List of terms in the query
            term_frequencies: Dictionary mapping terms to their frequency in the document
            doc_length: Length of the document (in tokens)
            term_doc_frequencies: Dictionary mapping terms to their document frequency

        Returns:
            BM25 score for the document
        """
        score = 0.0

        # Length normalization factor
        length_norm = 1 - self.b + self.b * (doc_length / self.avg_doc_length)

        # Count query term frequencies for multi-word queries
        query_term_counts: defaultdict[str, int] = defaultdict(int)
        for term in query_terms:
            query_term_counts[term] += 1

        for term, query_count in query_term_counts.items():
            if term not in term_frequencies:
                continue

            # Get term frequency in document
            tf = term_frequencies[term]

            # Get document frequency for IDF calculation
            df = term_doc_frequencies.get(term, 0)
            if df == 0:
                continue

            # Compute IDF
            idf = self.compute_idf(term, df)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * length_norm

            # Add term contribution to score (weighted by query term frequency)
            score += idf * (numerator / denominator) * query_count

        return score

    def score_documents_batch(
        self,
        query_terms: list[str],
        documents: list[tuple[int, dict[str, int], int]],
        term_doc_frequencies: dict[str, int],
    ) -> list[tuple[int, float]]:
        """Score multiple documents at once for better performance.

        Args:
            query_terms: List of terms in the query
            documents: List of (doc_id, term_frequencies, doc_length) tuples
            term_doc_frequencies: Dictionary mapping terms to their document frequency

        Returns:
            List of (doc_id, score) tuples
        """
        results = []

        for doc_id, term_freqs, doc_length in documents:
            score = self.score_document(query_terms, term_freqs, doc_length, term_doc_frequencies)
            if score > 0:
                results.append((doc_id, score))

        return results

    def score_document_with_postings(
        self,
        query_terms: list[str],
        term_postings: dict[str, list["Posting"]],
        doc_lengths: dict[int, int],
        term_doc_frequencies: dict[str, int],
    ) -> dict[int, float]:
        """Score documents using standard BM25 (no field weights).

        This method provides the same interface as BM25FScorer but ignores
        field information and treats all term occurrences equally.

        Args:
            query_terms: List of terms in the query
            term_postings: Dictionary mapping terms to their posting lists
            doc_lengths: Dictionary mapping doc_id to document length
            term_doc_frequencies: Dictionary mapping terms to document frequency

        Returns:
            Dictionary mapping doc_id to BM25 score
        """
        # Collect candidate documents and compute simple term frequencies
        doc_term_freqs: defaultdict[int, defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for term in query_terms:
            if term in term_postings:
                for posting in term_postings[term]:
                    # Count total occurrences across all fields (ignore field info)
                    # posting.positions is list of (field_id, position) tuples
                    # so length = number of occurrences
                    total_freq = len(posting.positions)
                    doc_term_freqs[posting.doc_id][term] = total_freq

        # Score each document using standard BM25
        doc_scores: dict[int, float] = {}
        for doc_id, term_freqs in doc_term_freqs.items():
            doc_length = doc_lengths.get(doc_id, self.avg_doc_length)
            score = self.score_document(
                query_terms=query_terms,
                term_frequencies=term_freqs,
                doc_length=doc_length,
                term_doc_frequencies=term_doc_frequencies,
            )
            if score > 0:
                doc_scores[doc_id] = score

        return doc_scores


class BM25Plus(BM25Scorer):
    """BM25+ variant that adds a small constant to prevent zero scores.

    BM25+ addresses a potential issue with BM25 where long documents with
    low term frequencies might get unfairly penalized.
    """

    def __init__(
        self,
        num_documents: int,
        avg_doc_length: float,
        k1: float = 1.2,
        b: float = 0.75,
        delta: float = 1.0,
    ):
        """Initialize BM25+ scorer.

        Args:
            num_documents: Total number of documents in the collection
            avg_doc_length: Average document length (in tokens)
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            delta: Small constant added to TF component (default: 1.0)
        """
        super().__init__(num_documents, avg_doc_length, k1, b)
        self.delta = delta

    def score_document(
        self,
        query_terms: list[str],
        term_frequencies: dict[str, int],
        doc_length: int | float,
        term_doc_frequencies: dict[str, int],
    ) -> float:
        """Calculate BM25+ score (includes delta term)."""
        score = 0.0
        length_norm = 1 - self.b + self.b * (doc_length / self.avg_doc_length)

        query_term_counts: defaultdict[str, int] = defaultdict(int)
        for term in query_terms:
            query_term_counts[term] += 1

        for term, query_count in query_term_counts.items():
            if term not in term_frequencies:
                continue

            tf = term_frequencies[term]
            df = term_doc_frequencies.get(term, 0)
            if df == 0:
                continue

            idf = self.compute_idf(term, df)

            # BM25+ formula: adds delta to the numerator
            numerator = (tf + self.delta) * (self.k1 + 1)
            denominator = tf + self.k1 * length_norm

            score += idf * (numerator / denominator) * query_count

        return score


class BM25FScorer(BM25Scorer):
    """BM25F: Field-weighted variant of BM25 for structured documents.

    This scorer applies different weights to term occurrences based on
    which field they appear in (title, heading, lead, body).

    Key features:
    - Title matches receive highest weight (default 3x)
    - Heading matches receive high weight (default 2x)
    - Lead paragraph matches receive high weight (default 2x)
    - Body matches receive baseline weight (1x)

    This significantly improves ranking for navigational queries where
    users search for document titles.
    """

    def __init__(
        self,
        num_documents: int,
        avg_doc_length: float,
        k1: float = 1.2,
        b: float = 0.75,
        field_weights: dict[int, float] | None = None,
    ):
        """Initialize BM25F scorer.

        Args:
            num_documents: Total number of documents in the collection
            avg_doc_length: Average document length (in tokens)
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            field_weights: Dictionary mapping FieldType values to boost weights.
                         If None, uses default weights:
                         {TITLE: 3.0, HEADING: 2.0, LEAD: 2.0, BODY: 1.0}
        """
        super().__init__(num_documents, avg_doc_length, k1, b)

        # Import here to avoid circular dependency
        from indexa.indexer.models import FieldType

        # Set default field weights if not provided
        if field_weights is None:
            self.field_weights = {
                int(FieldType.TITLE): 3.0,  # Title matches highly weighted
                int(FieldType.HEADING): 2.0,  # Heading matches moderately weighted
                int(FieldType.LEAD): 2.0,  # Lead paragraph moderately weighted
                int(FieldType.BODY): 1.0,  # Body matches baseline
            }
        else:
            self.field_weights = field_weights

    def score_document_with_postings(
        self,
        query_terms: list[str],
        term_postings: dict[str, list["Posting"]],
        doc_lengths: dict[int, int],
        term_doc_frequencies: dict[str, int],
    ) -> dict[int, float]:
        """Score documents using field-weighted BM25F.

        This method takes posting lists (with field information) instead of
        pre-computed term frequencies, allowing field-specific weighting.

        Args:
            query_terms: List of terms in the query
            term_postings: Dictionary mapping terms to their posting lists
            doc_lengths: Dictionary mapping doc_id to document length
            term_doc_frequencies: Dictionary mapping terms to document frequency

        Returns:
            Dictionary mapping doc_id to BM25F score
        """
        # Pre-process postings into dict for O(1) lookup instead of O(n) scan
        # This transforms term_postings from {term: [posting1, posting2, ...]}
        # to {term: {doc_id: posting}}
        term_posting_maps: dict[str, dict[int, "Posting"]] = {}
        candidate_docs: set[int] = set()

        for term in query_terms:
            if term in term_postings:
                posting_map: dict[int, "Posting"] = {}
                for posting in term_postings[term]:
                    posting_map[posting.doc_id] = posting
                    candidate_docs.add(posting.doc_id)
                term_posting_maps[term] = posting_map

        # Count query term frequencies for multi-word queries
        query_term_counts: defaultdict[str, int] = defaultdict(int)
        for term in query_terms:
            query_term_counts[term] += 1

        # Score each candidate document
        doc_scores: dict[int, float] = {}

        for doc_id in candidate_docs:
            score = 0.0
            doc_length = doc_lengths.get(doc_id, self.avg_doc_length)
            length_norm = 1 - self.b + self.b * (doc_length / self.avg_doc_length)

            for term, query_count in query_term_counts.items():
                if term not in term_posting_maps:
                    continue

                # O(1) lookup instead of O(n) scan
                posting = term_posting_maps[term].get(doc_id)
                if posting is None:
                    continue

                # Compute field-weighted term frequency
                field_counts: defaultdict[int, int] = defaultdict(int)
                for field_id, _ in posting.positions:
                    field_counts[field_id] += 1

                # Apply field weights
                weighted_tf = 0.0
                for field_id, count in field_counts.items():
                    weight = self.field_weights.get(field_id, 1.0)
                    weighted_tf += weight * count

                if weighted_tf == 0:
                    continue

                # Get document frequency for IDF calculation
                df = term_doc_frequencies.get(term, 0)
                if df == 0:
                    continue

                # Compute IDF
                idf = self.compute_idf(term, df)

                # BM25 formula with weighted TF
                numerator = weighted_tf * (self.k1 + 1)
                denominator = weighted_tf + self.k1 * length_norm

                # Add term contribution to score
                score += idf * (numerator / denominator) * query_count

            if score > 0:
                doc_scores[doc_id] = score

        return doc_scores
