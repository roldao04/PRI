"""Search engine implementation with BM25 ranking and relevance feedback.

This module provides the main SearchEngine class that orchestrates
query processing, ranking, and relevance feedback functionality.
"""

import logging
import math
from pathlib import Path

from indexa.indexer.models import DocumentInfo
from indexa.indexer.tokenizer import Tokenizer, TokenizerConfig
from indexa.searcher.bm25 import BM25FScorer
from indexa.searcher.document_store import DocumentStore
from indexa.searcher.index_loader import IndexLoader

logger = logging.getLogger(__name__)


class SearchResult:
    """Represents a single search result."""

    def __init__(
        self,
        doc_id: int,
        score: float,
        title: str,
        content_preview: str = "",
        url: str = "",
    ):
        """Initialize search result.

        Args:
            doc_id: Document ID
            score: BM25 relevance score
            title: Document title
            content_preview: Preview of document content
            url: URL to view document (for Wikipedia articles)
        """
        self.doc_id = doc_id
        self.score = score
        self.title = title
        self.content_preview = content_preview
        self.url = url

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "doc_id": self.doc_id,
            "score": round(self.score, 4),
            "title": self.title,
            "content_preview": self.content_preview,
            "url": self.url,
        }


class SimilarityCalculator:
    """Calculate document similarity using various metrics.

    Supports multiple similarity methods:
    - Cosine similarity with TF-IDF
    - Field-weighted similarity (using document structure)
    - Hybrid similarity (combining multiple metrics)
    - Legacy BM25-based similarity (for backward compatibility)

    NOTE: With lazy loading and incremental vector building, we use document
    lengths for proper normalization since vectors may be incomplete.
    """

    def __init__(self, index, num_documents: int):
        """Initialize similarity calculator.

        Args:
            index: The inverted index
            num_documents: Total number of documents in the collection
        """
        self.index = index
        self.num_documents = num_documents

        # Pre-compute average TF-IDF weight per term for normalization
        # This helps estimate full vector magnitudes from incomplete vectors
        self.avg_doc_length = (
            index.get_avg_doc_length() if hasattr(index, "get_avg_doc_length") else 100
        )

    def compute_cosine_similarity(
        self,
        doc_vector: dict[str, int],
        candidate_vectors: dict[int, dict[str, int]],
        min_score: float = 0.0,
        source_doc_id: int | None = None,
    ) -> list[tuple[int, float]]:
        """Compute TF-IDF similarity between source document and candidates.

        Uses unnormalized TF-IDF dot product (similar to BM25 philosophy).
        This produces unbounded scores (typically 0-50+) matching the range
        of BM25 scores from regular search.

        Args:
            doc_vector: Term frequency vector for source document
            candidate_vectors: Dictionary mapping doc_id to term frequency vectors
            min_score: Minimum similarity score threshold
            source_doc_id: Source document ID (unused, kept for API compatibility)

        Returns:
            List of (doc_id, similarity_score) tuples, sorted by score (unbounded)
        """
        # Compute TF-IDF vector for source document
        source_tfidf = self._compute_tfidf_vector(doc_vector)

        if not source_tfidf:
            logger.warning("Source document has no TF-IDF terms")
            return []

        # Compute similarities for all candidates
        similarities = []

        for doc_id, candidate_vector in candidate_vectors.items():
            # Compute TF-IDF for candidate (only shared terms)
            candidate_tfidf = self._compute_tfidf_vector(candidate_vector)

            if not candidate_tfidf:
                continue

            # Compute dot product (only for terms in both documents)
            # This is the unnormalized TF-IDF similarity score
            score = 0.0
            for term, weight in source_tfidf.items():
                if term in candidate_tfidf:
                    score += weight * candidate_tfidf[term]

            if score <= 0:
                continue

            if score >= min_score:
                similarities.append((doc_id, score))

        # Sort by score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            f"Computed TF-IDF similarity for {len(similarities)} candidates "
            f"(score range: {similarities[0][1]:.2f} to {similarities[-1][1]:.2f})"
            if similarities
            else "No candidates above threshold"
        )

        return similarities

    def compute_field_weighted_similarity(
        self,
        doc_id: int,
        candidate_docs: set[int],
        min_score: float = 0.0,
        field_weights: dict[str, float] | None = None,
    ) -> list[tuple[int, float]]:
        """Compute field-weighted similarity using document structure.

        This method computes separate similarities for each field (title, heading, lead, body)
        and combines them with configurable weights. Documents with similar titles and
        structure will rank higher.

        Args:
            doc_id: Source document ID
            candidate_docs: Set of candidate document IDs
            min_score: Minimum similarity score threshold
            field_weights: Optional custom weights
                (default: title=3.0, heading=2.0, lead=2.0, body=1.0)

        Returns:
            List of (doc_id, similarity_score) tuples, sorted by score
        """
        # Default field weights (matching BM25F defaults)
        if field_weights is None:
            field_weights = {
                "title": 3.0,
                "heading": 2.0,
                "lead": 2.0,
                "body": 1.0,
            }

        from indexa.indexer.models import FieldType

        # Get field-specific vectors for source document
        source_field_vectors = self._get_field_vectors(doc_id)

        # Compute weighted similarities
        similarities = []

        for candidate_id in candidate_docs:
            if candidate_id == doc_id:
                continue

            # Get field vectors for candidate
            candidate_field_vectors = self._get_field_vectors(candidate_id)

            # Compute similarity for each field
            total_similarity = 0.0
            total_weight = 0.0

            for field_type in FieldType:
                field_name = field_type.name.lower()
                weight = field_weights.get(field_name, 1.0)

                source_vector = source_field_vectors.get(field_type.value, {})
                candidate_vector = candidate_field_vectors.get(field_type.value, {})

                if not source_vector or not candidate_vector:
                    continue

                # Compute cosine similarity for this field
                field_similarity = self._cosine_similarity_vectors(source_vector, candidate_vector)

                total_similarity += field_similarity * weight
                total_weight += weight

            # Normalize by total weight
            if total_weight > 0:
                final_similarity = total_similarity / total_weight

                if final_similarity >= min_score:
                    similarities.append((candidate_id, final_similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities

    def compute_jaccard_similarity(
        self,
        doc_vector: dict[str, int],
        candidate_vectors: dict[int, dict[str, int]],
        min_score: float = 0.0,
    ) -> list[tuple[int, float]]:
        """Compute Jaccard similarity based on term overlap.

        Jaccard = |intersection| / |union|
        Useful for finding documents with similar vocabulary.

        Args:
            doc_vector: Term frequency vector for source document
            candidate_vectors: Dictionary mapping doc_id to term frequency vectors
            min_score: Minimum similarity score threshold

        Returns:
            List of (doc_id, similarity_score) tuples, sorted by score
        """
        source_terms = set(doc_vector.keys())
        similarities = []

        for doc_id, candidate_vector in candidate_vectors.items():
            candidate_terms = set(candidate_vector.keys())

            intersection = len(source_terms & candidate_terms)
            union = len(source_terms | candidate_terms)

            if union == 0:
                continue

            similarity = intersection / union

            if similarity >= min_score:
                similarities.append((doc_id, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities

    def compute_hybrid_similarity(
        self,
        doc_vector: dict[str, int],
        candidate_vectors: dict[int, dict[str, int]],
        min_score: float = 0.0,
        tfidf_weight: float = 0.7,
        jaccard_weight: float = 0.3,
        source_doc_id: int | None = None,
    ) -> list[tuple[int, float]]:
        """Compute hybrid similarity combining TF-IDF and Jaccard metrics.

        Combines TF-IDF dot product (unbounded) with Jaccard similarity ([0,1]).
        Jaccard is scaled to match TF-IDF range before combining.

        Args:
            doc_vector: Term frequency vector for source document
            candidate_vectors: Dictionary mapping doc_id to term frequency vectors
            min_score: Minimum similarity score threshold
            tfidf_weight: Weight for TF-IDF score (default: 0.7)
            jaccard_weight: Weight for Jaccard score (default: 0.3)
            source_doc_id: Source document ID (unused, kept for compatibility)

        Returns:
            List of (doc_id, similarity_score) tuples, sorted by score (unbounded)
        """
        # Compute both similarities
        tfidf_scores = dict(
            self.compute_cosine_similarity(
                doc_vector, candidate_vectors, min_score=0.0, source_doc_id=source_doc_id
            )
        )
        jaccard_scores = dict(
            self.compute_jaccard_similarity(doc_vector, candidate_vectors, min_score=0.0)
        )

        # Find max TF-IDF score for scaling
        max_tfidf = max(tfidf_scores.values()) if tfidf_scores else 1.0

        # Combine scores (scale Jaccard to TF-IDF range)
        all_docs = set(tfidf_scores.keys()) | set(jaccard_scores.keys())
        similarities = []

        for doc_id in all_docs:
            tfidf = tfidf_scores.get(doc_id, 0.0)
            jaccard = jaccard_scores.get(doc_id, 0.0)

            # Scale Jaccard [0,1] to TF-IDF range [0, max_tfidf]
            scaled_jaccard = jaccard * max_tfidf

            # Weighted combination
            combined = (tfidf * tfidf_weight) + (scaled_jaccard * jaccard_weight)

            if combined >= min_score:
                similarities.append((doc_id, combined))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            f"Computed hybrid similarity: max_tfidf={max_tfidf:.2f}, "
            f"{len(similarities)} candidates "
            f"(score range: {similarities[0][1]:.2f} to {similarities[-1][1]:.2f})"
            if similarities
            else "No candidates above threshold"
        )

        return similarities

    def _compute_tfidf_vector(self, term_freqs: dict[str, int]) -> dict[str, float]:
        """Compute TF-IDF weights for a term frequency vector.

        Args:
            term_freqs: Dictionary mapping terms to frequencies

        Returns:
            Dictionary mapping terms to TF-IDF weights
        """
        tfidf = {}

        for term, tf in term_freqs.items():
            df = self.index.get_doc_frequency(term)
            if df == 0:
                continue

            # TF-IDF: (1 + log(tf)) * log(N / df)
            tf_weight = 1 + math.log(tf) if tf > 0 else 0
            idf_weight = math.log(self.num_documents / df) if df > 0 else 0

            tfidf[term] = tf_weight * idf_weight

        return tfidf

    def _cosine_similarity_vectors(self, vector1: dict[str, int], vector2: dict[str, int]) -> float:
        """Compute cosine similarity between two term frequency vectors.

        Args:
            vector1: First term frequency vector
            vector2: Second term frequency vector

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        # Compute TF-IDF vectors
        tfidf1 = self._compute_tfidf_vector(vector1)
        tfidf2 = self._compute_tfidf_vector(vector2)

        # Compute magnitudes
        mag1 = math.sqrt(sum(w * w for w in tfidf1.values()))
        mag2 = math.sqrt(sum(w * w for w in tfidf2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        # Compute dot product
        dot_product = sum(tfidf1[term] * tfidf2.get(term, 0.0) for term in tfidf1)

        return dot_product / (mag1 * mag2)

    def _get_field_vectors(self, doc_id: int) -> dict[int, dict[str, int]]:
        """Get term frequency vectors separated by field.

        Args:
            doc_id: Document ID

        Returns:
            Dictionary mapping field_id to term frequency vectors
        """
        field_vectors: dict[int, dict[str, int]] = {}

        # Iterate through entire index to find all terms in this document
        for term, postings in self.index.index.items():
            for posting in postings:
                if posting.doc_id == doc_id:
                    # Group positions by field
                    for field_id, position in posting.positions:
                        if field_id not in field_vectors:
                            field_vectors[field_id] = {}

                        if term not in field_vectors[field_id]:
                            field_vectors[field_id][term] = 0

                        field_vectors[field_id][term] += 1

                    break

        return field_vectors


class SearchEngine:
    """Main search engine with BM25 ranking and relevance feedback.

    This class provides the complete search functionality including:
    - Query processing with tokenization
    - BM25 ranking
    - Relevance feedback (finding similar documents)
    - Real-time performance (< 1 second)
    """

    def __init__(
        self,
        index_dir: Path,
        k1: float = 1.2,
        b: float = 0.75,
        lazy_loading: bool = False,
        use_field_weights: bool = True,
    ):
        """Initialize search engine.

        Args:
            index_dir: Directory containing the index files
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
            lazy_loading: Whether to use lazy loading for the index
            use_field_weights: Whether to use BM25F with field weights (default: True)
        """
        self.index_dir = Path(index_dir)
        self.k1 = k1
        self.b = b
        self.use_field_weights = use_field_weights

        # Load index
        logger.info("Loading index...")
        loader = IndexLoader(index_dir)

        if lazy_loading:
            self.index = loader.load_lazy()
        else:
            self.index = loader.load()

        logger.info("Index loaded successfully")

        # Initialize document store for retrieving full content
        # Look for the SQLite database in the index directory
        db_file = index_dir / "documents.db"
        if db_file.exists():
            self.document_store = DocumentStore(db_file)
            logger.info(f"Initialized document store with SQLite: {db_file}")
        else:
            logger.warning(
                "SQLite database (documents.db) not found in index directory, "
                "document content retrieval disabled. Re-index to create the database."
            )
            self.document_store = None

        # Initialize tokenizer with same config as indexer
        if self.index.metadata:
            tokenizer_config = TokenizerConfig.from_dict(self.index.metadata.tokenizer_config)
            self.tokenizer = Tokenizer(tokenizer_config)
            logger.info(f"Initialized tokenizer with config: {tokenizer_config}")
        else:
            # Fallback to default config
            logger.warning("No tokenizer config found in metadata, using defaults")
            self.tokenizer = Tokenizer(TokenizerConfig())

        # Initialize BM25 scorer (with or without field weights)
        num_docs = self.index.get_num_documents()
        avg_doc_len = self.index.get_avg_doc_length()

        if self.use_field_weights:
            self.bm25_scorer = BM25FScorer(
                num_documents=num_docs,
                avg_doc_length=avg_doc_len,
                k1=k1,
                b=b,
                # Use default field weights: TITLE=3.0, HEADING=2.0, LEAD=2.0, BODY=1.0
            )
            logger.info(
                f"Initialized BM25F scorer (field-weighted): {num_docs} docs, "
                f"avg_len={avg_doc_len:.2f}, k1={k1}, b={b}"
            )
        else:
            from indexa.searcher.bm25 import BM25Scorer

            self.bm25_scorer = BM25Scorer(
                num_documents=num_docs,
                avg_doc_length=avg_doc_len,
                k1=k1,
                b=b,
            )
            logger.info(
                f"Initialized BM25 scorer (standard, no field weights): {num_docs} docs, "
                f"avg_len={avg_doc_len:.2f}, k1={k1}, b={b}"
            )

    def search(
        self,
        query: str,
        num_results: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search for documents matching a query.

        Args:
            query: The search query
            num_results: Maximum number of results to return
            min_score: Minimum BM25 score threshold

        Returns:
            List of SearchResult objects, sorted by relevance
        """
        logger.info(f"Searching for: {query}")

        # Tokenize query using same rules as indexer
        # Extract just the terms (ignore positions)
        query_terms = [term for term, _ in self.tokenizer.tokenize_with_positions(query)]

        if not query_terms:
            logger.warning("Query produced no valid tokens after tokenization")
            return []

        logger.debug(f"Query tokens: {query_terms}")

        # Get candidate documents (documents containing at least one query term)
        candidate_docs = self._get_candidate_documents(query_terms)

        if not candidate_docs:
            logger.info("No documents found containing query terms")
            return []

        logger.debug(f"Found {len(candidate_docs)} candidate documents")

        # Score all candidate documents
        scored_docs = self._score_documents(query_terms, candidate_docs)

        # Filter by minimum score
        if min_score > 0:
            scored_docs = [(doc_id, score) for doc_id, score in scored_docs if score >= min_score]

        # Sort by score (descending) and take top N
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = scored_docs[:num_results]

        # Convert to SearchResult objects
        results = self._build_search_results(top_docs)

        logger.info(f"Returning {len(results)} results")
        return results

    def search_similar(
        self,
        doc_id: int,
        num_results: int = 10,
        min_score: float = 0.0,
        similarity_method: str = "hybrid",
    ) -> list[SearchResult]:
        """Find documents similar to a given document (relevance feedback).

        This method supports multiple similarity algorithms:
        - 'hybrid': Combines cosine and Jaccard (70/30 weight) - DEFAULT, RECOMMENDED
        - 'cosine': TF-IDF cosine similarity (fast, good quality)
        - 'field-weighted': Field-aware cosine similarity
            (uses title/heading/lead/body, may be slow on first use)
        - 'jaccard': Jaccard coefficient (set-based similarity, very fast)
        - 'bm25': Legacy BM25-based method (for backward compatibility)

        Args:
            doc_id: ID of the document to find similar documents for
            num_results: Maximum number of results to return
            min_score: Minimum similarity score threshold
            similarity_method: Similarity algorithm to use (default: 'hybrid')

        Returns:
            List of SearchResult objects, sorted by similarity
        """
        import time

        start_time = time.time()
        logger.info(
            f"Finding documents similar to doc_id={doc_id} using method='{similarity_method}'"
        )

        # Check if we're using lazy loading (affects strategy)
        is_lazy = hasattr(self.index, "term_offsets")

        # Get the document vector (term frequencies)
        # For lazy loading, use fast approximate vector; otherwise use complete vector
        if is_lazy and similarity_method != "field-weighted":
            logger.debug("Using fast approximate vector for lazy loading")
            doc_vector = self._get_approximate_document_vector_lazy(doc_id, max_terms=1000)
        else:
            doc_vector = self.index.get_document_vector(doc_id)

        if not doc_vector:
            logger.warning(f"Document {doc_id} not found or empty")
            return []

        logger.debug(f"Source document vector has {len(doc_vector)} terms")

        # Legacy BM25-based method (for backward compatibility)
        if similarity_method == "bm25":
            return self._search_similar_bm25(doc_id, doc_vector, num_results, min_score)

        # Get candidate documents (documents sharing at least one term)
        candidate_docs_info = self._get_candidate_documents_for_similarity(doc_vector)

        # Remove the source document from candidates
        candidate_docs_info = {
            did: vec for did, vec in candidate_docs_info.items() if did != doc_id
        }

        if not candidate_docs_info:
            logger.info("No candidate documents found")
            return []

        logger.debug(f"Found {len(candidate_docs_info)} candidate documents")

        # Initialize similarity calculator
        similarity_calc = SimilarityCalculator(self.index, self.index.get_num_documents())

        # Compute similarities based on selected method
        if similarity_method == "cosine":
            scored_docs = similarity_calc.compute_cosine_similarity(
                doc_vector, candidate_docs_info, min_score, source_doc_id=doc_id
            )
        elif similarity_method == "field-weighted":
            # For field-weighted, we need candidate doc IDs, not vectors
            candidate_ids = set(candidate_docs_info.keys())
            scored_docs = similarity_calc.compute_field_weighted_similarity(
                doc_id, candidate_ids, min_score
            )
        elif similarity_method == "jaccard":
            scored_docs = similarity_calc.compute_jaccard_similarity(
                doc_vector, candidate_docs_info, min_score
            )
        elif similarity_method == "hybrid":
            scored_docs = similarity_calc.compute_hybrid_similarity(
                doc_vector, candidate_docs_info, min_score, source_doc_id=doc_id
            )
        else:
            logger.warning(
                f"Unknown similarity method: {similarity_method}, falling back to cosine"
            )
            scored_docs = similarity_calc.compute_cosine_similarity(
                doc_vector, candidate_docs_info, min_score, source_doc_id=doc_id
            )

        # Take top N results
        top_docs = scored_docs[:num_results]

        # Build results
        results = self._build_search_results(top_docs)

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            f"Returning {len(results)} similar documents "
            f"(method={similarity_method}, time={elapsed:.2f}ms)"
        )
        return results

    def _search_similar_bm25(
        self,
        doc_id: int,
        doc_vector: dict[str, int],
        num_results: int,
        min_score: float,
    ) -> list[SearchResult]:
        """Legacy BM25-based similarity search (for backward compatibility).

        Uses the old approach: extract top 50 terms by TF-IDF and use as query.

        Args:
            doc_id: Source document ID
            doc_vector: Term frequency vector for source document
            num_results: Maximum number of results
            min_score: Minimum score threshold

        Returns:
            List of SearchResult objects
        """
        # Use the most important terms from the document as a pseudo-query
        # Weight by TF-IDF to get the most discriminative terms
        term_weights = self._compute_tfidf_weights(doc_vector)

        # Take top terms by TF-IDF weight
        top_terms = sorted(term_weights.items(), key=lambda x: x[1], reverse=True)[:50]
        query_terms = [term for term, _ in top_terms]

        logger.debug(f"Using {len(query_terms)} top terms for BM25 similarity search")

        # Get candidate documents
        candidate_docs = self._get_candidate_documents(query_terms)

        # Remove the source document from candidates
        candidate_docs = {did: info for did, info in candidate_docs.items() if did != doc_id}

        if not candidate_docs:
            return []

        # Score documents using BM25
        scored_docs = self._score_documents(query_terms, candidate_docs)

        # Filter and sort
        if min_score > 0:
            scored_docs = [(did, score) for did, score in scored_docs if score >= min_score]

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        top_docs = scored_docs[:num_results]

        # Build results
        results = self._build_search_results(top_docs)

        return results

    def _get_candidate_documents(self, query_terms: list[str]) -> dict[int, DocumentInfo]:
        """Get all documents containing at least one query term.

        Args:
            query_terms: List of query terms

        Returns:
            Dictionary mapping doc_id to DocumentInfo
        """
        candidate_docs: dict[int, DocumentInfo] = {}

        for term in query_terms:
            postings = self.index.get_postings(term)

            for posting in postings:
                if posting.doc_id not in candidate_docs:
                    doc_info = self.index.get_document_info(posting.doc_id)
                    if doc_info:
                        candidate_docs[posting.doc_id] = doc_info

        return candidate_docs

    def _get_approximate_document_vector_lazy(
        self,
        doc_id: int,
        max_terms: int = 1000,
    ) -> dict[str, int]:
        """Build approximate document vector for lazy loading (FAST).

        For lazy loading, building a complete vector requires scanning the entire
        vocabulary (very slow). Instead, we build an approximate vector using:
        1. Terms already in cache (free)
        2. Sample/rank by importance and take top K terms

        This is much faster and works well for similarity since we only need
        discriminative terms, not every single term.

        Args:
            doc_id: Document ID
            max_terms: Maximum number of terms to include (default: 1000)

        Returns:
            Approximate term frequency vector (contains most important terms)
        """
        doc_vector: dict[str, int] = {}

        # Collect terms from cache (already loaded, no disk I/O)
        for term, postings in self.index.index.items():
            for posting in postings:
                if posting.doc_id == doc_id:
                    doc_vector[term] = len(posting.positions)
                    break

        logger.debug(
            f"Built approximate vector for doc_id={doc_id} " f"from {len(doc_vector)} cached terms"
        )

        # If we have very few cached terms, intelligently sample from vocabulary
        # to build a more representative vector (still much faster than full scan)
        if len(doc_vector) < 50:
            logger.debug(
                f"Only {len(doc_vector)} cached terms for doc_id={doc_id}. "
                f"Sampling vocabulary to build representative vector..."
            )

            # Sample every Nth term from vocabulary (systematic sampling)
            # This is much faster than full scan but gives representative sample
            if hasattr(self.index, "term_offsets"):
                all_terms = list(self.index.term_offsets.keys())  # type: ignore
                sample_rate = max(1, len(all_terms) // 2000)  # Sample ~2000 terms
                sample_terms = all_terms[::sample_rate]

                logger.debug(f"Sampling {len(sample_terms)} terms (every {sample_rate}th term)")

                for term in sample_terms:
                    if len(doc_vector) >= max_terms:
                        break  # Stop if we have enough

                    postings = self.index.get_postings(term)
                    for posting in postings:
                        if posting.doc_id == doc_id:
                            doc_vector[term] = len(posting.positions)
                            break

                logger.debug(
                    f"After sampling: vector has {len(doc_vector)} terms for doc_id={doc_id}"
                )

        # Rank by TF-IDF and take top max_terms
        if len(doc_vector) > max_terms:
            # Compute TF-IDF scores
            term_scores = self._compute_tfidf_weights(doc_vector)
            # Sort by score and take top K
            top_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:max_terms]
            doc_vector = {term: doc_vector[term] for term, _ in top_terms}
            logger.debug(f"Sampled top {len(doc_vector)} terms by TF-IDF")

        return doc_vector

    def _get_candidate_documents_for_similarity(
        self, source_vector: dict[str, int]
    ) -> dict[int, dict[str, int]]:
        """Get candidate documents with their term vectors for similarity calculation.

        OPTIMIZED FOR LAZY LOADING: Builds candidate vectors incrementally from
        postings as we load them, avoiding expensive get_document_vector() calls.

        Note: With lazy loading, this builds APPROXIMATE vectors containing only
        terms seen during the search. This is much faster and works well for
        similarity since we're primarily interested in shared terms anyway.

        Args:
            source_vector: Term frequency vector for the source document

        Returns:
            Dictionary mapping doc_id to term frequency vectors
            (may be incomplete with lazy loading)
        """
        from collections import defaultdict

        # Build candidate vectors incrementally as we load postings
        # This avoids calling get_document_vector() N times with lazy loading
        candidate_vectors: dict[int, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))

        # For each term in source document, load its postings and update candidate vectors
        for term in source_vector:
            postings = self.index.get_postings(term)

            for posting in postings:
                doc_id = posting.doc_id
                # Count term frequency (number of positions)
                term_freq = len(posting.positions)
                candidate_vectors[doc_id][term] = term_freq

        # Convert defaultdicts to regular dicts
        return {doc_id: dict(vector) for doc_id, vector in candidate_vectors.items()}

    def _score_documents(
        self,
        query_terms: list[str],
        candidate_docs: dict[int, DocumentInfo],
    ) -> list[tuple[int, float]]:
        """Score candidate documents using BM25F (field-weighted).

        Args:
            query_terms: List of query terms
            candidate_docs: Dictionary of candidate documents

        Returns:
            List of (doc_id, score) tuples
        """
        import time

        start = time.time()

        # Build term document frequencies for query terms
        term_doc_frequencies = {term: self.index.get_doc_frequency(term) for term in query_terms}

        # Get posting lists for query terms (needed for field-weighted scoring)
        term_postings = {}
        for term in query_terms:
            postings = self.index.get_postings(term)
            if postings:
                term_postings[term] = postings

        load_time = time.time() - start
        logger.debug(f"Loaded postings in {load_time*1000:.2f}ms")

        # Build document length map
        doc_lengths = {doc_id: doc_info.length for doc_id, doc_info in candidate_docs.items()}

        # Score all documents using BM25F
        score_start = time.time()
        doc_scores = self.bm25_scorer.score_document_with_postings(
            query_terms=query_terms,
            term_postings=term_postings,
            doc_lengths=doc_lengths,
            term_doc_frequencies=term_doc_frequencies,
        )
        score_time = time.time() - score_start
        logger.debug(f"BM25 scoring took {score_time*1000:.2f}ms for {len(candidate_docs)} docs")

        # Convert to list of tuples
        scored_docs = list(doc_scores.items())

        return scored_docs

    def _compute_tfidf_weights(self, doc_vector: dict[str, int]) -> dict[str, float]:
        """Compute TF-IDF weights for terms in a document.

        Args:
            doc_vector: Dictionary mapping terms to frequencies

        Returns:
            Dictionary mapping terms to TF-IDF weights
        """
        weights = {}
        num_docs = self.index.get_num_documents()

        for term, tf in doc_vector.items():
            df = self.index.get_doc_frequency(term)
            if df == 0:
                continue

            # TF-IDF: (1 + log(tf)) * log(N / df)
            tf_weight = 1 + math.log(tf) if tf > 0 else 0
            idf_weight = math.log(num_docs / df) if df > 0 else 0

            weights[term] = tf_weight * idf_weight

        return weights

    def _build_search_results(self, scored_docs: list[tuple[int, float]]) -> list[SearchResult]:
        """Build SearchResult objects from scored documents.

        Args:
            scored_docs: List of (doc_id, score) tuples

        Returns:
            List of SearchResult objects
        """
        results = []

        for doc_id, score in scored_docs:
            doc_info = self.index.get_document_info(doc_id)
            if not doc_info:
                continue

            # Create Wikipedia URL if applicable
            url = ""
            if "wiki" in doc_info.file_path.lower():
                # Title might need URL encoding, but for now keep it simple
                encoded_title = doc_info.title.replace(" ", "_")
                url = f"https://pt.wikipedia.org/wiki/{encoded_title}"

            # Get content preview from document store if available
            content_preview = doc_info.title
            if self.document_store:
                preview = self.document_store.get_document_preview(doc_id, max_length=200)
                if preview:
                    content_preview = preview

            result = SearchResult(
                doc_id=doc_id,
                score=score,
                title=doc_info.title,
                content_preview=content_preview,
                url=url,
            )

            results.append(result)

        return results

    def get_document_info(self, doc_id: int) -> DocumentInfo | None:
        """Get information about a specific document.

        Args:
            doc_id: Document ID

        Returns:
            DocumentInfo object or None if not found
        """
        return self.index.get_document_info(doc_id)

    def get_full_document(self, doc_id: int) -> dict | None:
        """Get full document content including title, text, and URL.

        Args:
            doc_id: Document ID

        Returns:
            Dictionary with 'title', 'content', 'url' or None if not found
        """
        if self.document_store:
            return self.document_store.get_document(doc_id)
        return None

    def get_statistics(self) -> dict:
        """Get index statistics.

        Returns:
            Dictionary with index statistics
        """
        stats = {
            "num_documents": self.index.get_num_documents(),
            "num_terms": len(self.index.get_all_terms()) if hasattr(self.index, "index") else 0,
            "avg_doc_length": self.index.get_avg_doc_length(),
            "bm25_k1": self.k1,
            "bm25_b": self.b,
        }

        if self.index.metadata:
            stats["metadata"] = {
                "num_terms_indexed": self.index.metadata.num_terms,
                "num_blocks": self.index.metadata.num_blocks,
            }

        return stats
