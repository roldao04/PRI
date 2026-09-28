"""Search endpoints."""

import logging
import time
from pathlib import Path as PathLib

from fastapi import APIRouter, HTTPException, Path, Query

from indexa.core.model import Document
from indexa.entrypoints.api.model import IndexStatsResponse, SearchResponse
from indexa.searcher.search_engine import SearchEngine

router = APIRouter(tags=["search engine"])
logger = logging.getLogger(__name__)

# Global search engine instance (initialized on first request)
_search_engine: SearchEngine | None = None


def get_search_engine() -> SearchEngine:
    """Get or initialize the search engine singleton.

    Returns:
        SearchEngine instance

    Raises:
        HTTPException: If search engine cannot be initialized
    """
    global _search_engine

    if _search_engine is None:
        try:
            # Try to find index in wiki_index_test_short directory
            index_dir = PathLib("wiki_index_full")
            if not index_dir.exists():
                raise FileNotFoundError(f"Index directory not found: {index_dir}")

            logger.info(f"Initializing search engine from {index_dir}")
            _search_engine = SearchEngine(
                index_dir=index_dir,
                k1=1.2,  # Default BM25 parameters
                b=0.75,
                lazy_loading=True,  # Use lazy loading for large datasets
                use_field_weights=False,  # Set to False to disable BM25F field weighting
            )
            logger.info("Search engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize search engine: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize search engine: {str(e)}",
            ) from e

    return _search_engine


@router.get("/search")
def search(
    query: str = Query(..., description="Search query", min_length=1),
    num_results: int = Query(10, description="Number of results to return", ge=1, le=100),
    min_score: float = Query(0.0, description="Minimum BM25 score threshold", ge=0.0),
    k1: float = Query(1.2, description="BM25 k1 parameter", ge=0.0, le=3.0),
    b: float = Query(0.75, description="BM25 b parameter", ge=0.0, le=1.0),
) -> SearchResponse:
    """Search for documents matching the given query using BM25 ranking.

    Args:
        query: The search query string
        num_results: Maximum number of results to return (1-100)
        min_score: Minimum BM25 score threshold (default: 0.0)
        k1: BM25 k1 parameter for term frequency saturation (default: 1.2)
        b: BM25 b parameter for length normalization (default: 0.75)

    Returns:
        SearchResponse with ranked results
    """
    start_time = time.time()

    try:
        engine = get_search_engine()

        # Update BM25 parameters if different from defaults
        if k1 != engine.k1 or b != engine.b:
            engine.bm25_scorer.k1 = k1
            engine.bm25_scorer.b = b
            engine.k1 = k1
            engine.b = b

        # Perform search
        search_results = engine.search(
            query=query,
            num_results=num_results,
            min_score=min_score,
        )

        # Convert to API Document model
        documents = [
            Document(
                id=result.doc_id,
                title=result.title,
                content=result.content_preview,
                score=result.score,
                url=result.url,
            )
            for result in search_results
        ]

        execution_time = (time.time() - start_time) * 1000  # Convert to ms

        logger.info(f"Query '{query}' returned {len(documents)} results in {execution_time:.2f}ms")

        return SearchResponse(
            results=documents,
            query=query,
            num_results=len(documents),
            execution_time_ms=round(execution_time, 2),
        )

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}") from e


@router.get("/search_similar")
def search_similar(
    doc_id: int = Query(..., description="Document ID to find similar documents for", ge=0),
    num_results: int = Query(10, description="Number of results to return", ge=1, le=100),
    min_score: float = Query(0.0, description="Minimum similarity score threshold", ge=0.0),
    similarity_method: str = Query(
        "hybrid",
        description=(
            "Similarity algorithm: 'cosine', 'field-weighted', 'jaccard', "
            "'hybrid', or 'bm25' (default: hybrid)"
        ),
        regex="^(cosine|field-weighted|jaccard|hybrid|bm25)$",
    ),
) -> SearchResponse:
    """Search for documents similar to the given document (relevance feedback).

    This endpoint implements relevance feedback by finding documents that are
    similar to a specified document. It's useful for "more like this" functionality.

    Supported similarity methods:
    - 'cosine': TF-IDF cosine similarity (default, recommended)
    - 'field-weighted': Field-aware cosine similarity (uses title/heading/lead/body structure)
    - 'jaccard': Jaccard coefficient (set-based similarity)
    - 'hybrid': Combines cosine and Jaccard (70/30 weight)
    - 'bm25': Legacy BM25-based method (for backward compatibility)

    Args:
        doc_id: The ID of the document to find similar documents for
        num_results: Maximum number of results to return (1-100)
        min_score: Minimum similarity score threshold (default: 0.0)
        similarity_method: Similarity algorithm to use (default: 'cosine')

    Returns:
        SearchResponse with similar documents ranked by similarity
    """
    start_time = time.time()

    try:
        engine = get_search_engine()

        # Verify document exists
        doc_info = engine.get_document_info(doc_id)
        if not doc_info:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID {doc_id} not found",
            )

        # Find similar documents
        search_results = engine.search_similar(
            doc_id=doc_id,
            num_results=num_results,
            min_score=min_score,
            similarity_method=similarity_method,
        )

        # Convert to API Document model
        documents = [
            Document(
                id=result.doc_id,
                title=result.title,
                content=result.content_preview,
                score=result.score,
                url=result.url,
            )
            for result in search_results
        ]

        execution_time = (time.time() - start_time) * 1000

        logger.info(
            f"Similar to doc_id={doc_id} returned {len(documents)} results "
            f"using method={similarity_method} in {execution_time:.2f}ms"
        )

        return SearchResponse(
            results=documents,
            query=f"Similar to: {doc_info.title} (method: {similarity_method})",
            num_results=len(documents),
            execution_time_ms=round(execution_time, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similar search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Similar search failed: {str(e)}") from e


@router.get("/document/{doc_id}")
def get_document(
    doc_id: int = Path(..., description="Document ID", ge=0),
) -> Document:
    """Get information about a specific document.

    Args:
        doc_id: The document ID

    Returns:
        Document information
    """
    try:
        engine = get_search_engine()

        # Try to get full document content first
        full_doc = engine.get_full_document(doc_id)

        if full_doc:
            # Return full content from Arrow file
            return Document(
                id=doc_id,
                title=full_doc["title"],
                content=full_doc["content"],
                url=full_doc["url"],
                score=0.0,
            )

        # Fallback to document info if full content not available
        doc_info = engine.get_document_info(doc_id)

        if not doc_info:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID {doc_id} not found",
            )

        # Create URL for Wikipedia
        url = ""
        if "wiki" in doc_info.file_path.lower():
            encoded_title = doc_info.title.replace(" ", "_")
            url = f"https://pt.wikipedia.org/wiki/{encoded_title}"

        return Document(
            id=doc_info.doc_id,
            title=doc_info.title,
            content=f"Document length: {doc_info.length} tokens (full content not available)",
            url=url,
            score=0.0,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get document error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}") from e


@router.get("/stats")
def get_stats() -> IndexStatsResponse:
    """Get statistics about the search index.

    Returns:
        Index statistics including number of documents, terms, and BM25 parameters
    """
    try:
        engine = get_search_engine()
        stats = engine.get_statistics()

        return IndexStatsResponse(
            num_documents=stats["num_documents"],
            num_terms=stats.get("num_terms", 0),
            avg_doc_length=stats["avg_doc_length"],
            bm25_k1=stats["bm25_k1"],
            bm25_b=stats["bm25_b"],
        )

    except Exception as e:
        logger.error(f"Get stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}") from e
