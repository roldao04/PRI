"""Search endpoints."""

import logging
import time
from pathlib import Path as PathLib
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Path, Query

from indexa.core.model import Document
from indexa.entrypoints.api.model import AnswerSnippet, IndexStatsResponse, SearchResponse
from indexa.searcher.search_engine import SearchEngine

if TYPE_CHECKING:
    from indexa.answer_generator.gemini_generator import GeminiAnswerGenerator
    from indexa.query_expansion.gemini_expander import GeminiQueryExpander
    from indexa.reranker.neural_reranker import NeuralReranker
    from indexa.snippet.snippet_extractor import SemanticSnippetExtractor

router = APIRouter(tags=["search engine"])
logger = logging.getLogger(__name__)

# Global search engine instance (initialized on first request)
_search_engine: SearchEngine | None = None

# Global reranker instances (model warming - initialized on first use per model)
_reranker_cache: dict[str, "NeuralReranker"] = {}

# Global answer generator instance (initialized on first use)
_answer_generator: "GeminiAnswerGenerator | None" = None

# Global query expander instance (initialized on first use)
_query_expander: "GeminiQueryExpander | None" = None

# Global snippet extractor cache (one per reranker model)
_snippet_extractor_cache: dict[str, "SemanticSnippetExtractor"] = {}


def get_reranker(model_name: str) -> "NeuralReranker":
    """Get or initialize a reranker singleton for the specified model.

    This implements model warming - the reranker is loaded once and reused
    across requests to avoid the overhead of loading the model each time.

    Args:
        model_name: Reranker model name ('unicamp-dl' or 'mmarco')

    Returns:
        NeuralReranker instance

    Raises:
        HTTPException: If reranker cannot be initialized
    """
    global _reranker_cache

    if model_name not in _reranker_cache:
        try:
            logger.info(f"Initializing reranker model: {model_name}")
            from indexa.reranker.neural_reranker import NeuralReranker

            _reranker_cache[model_name] = NeuralReranker(model_name=model_name)
            logger.info(f"Reranker {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to initialize reranker {model_name}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize reranker: {str(e)}",
            ) from e

    return _reranker_cache[model_name]


def get_answer_generator() -> "GeminiAnswerGenerator":
    """Get or initialize the answer generator singleton.

    Returns:
        GeminiAnswerGenerator instance

    Raises:
        HTTPException: If answer generator cannot be initialized
    """
    global _answer_generator

    if _answer_generator is None:
        try:
            logger.info("Initializing Gemini answer generator")
            from indexa.answer_generator.gemini_generator import GeminiAnswerGenerator

            _answer_generator = GeminiAnswerGenerator()
            logger.info("Answer generator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize answer generator: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize answer generator: {str(e)}. "
                "Make sure GEMINI_API_KEY is set in environment variables.",
            ) from e

    return _answer_generator


def get_snippet_extractor(reranker_model: str = "unicamp-dl") -> "SemanticSnippetExtractor":
    """Get or initialize the snippet extractor for a given reranker model.

    The snippet extractor reuses the reranker model to avoid loading models twice.

    Args:
        reranker_model: Reranker model name ('unicamp-dl' or 'mmarco')

    Returns:
        SemanticSnippetExtractor instance

    Raises:
        HTTPException: If snippet extractor cannot be initialized
    """
    global _snippet_extractor_cache

    if reranker_model not in _snippet_extractor_cache:
        try:
            logger.info(f"Initializing snippet extractor with model: {reranker_model}")
            from indexa.snippet.snippet_extractor import SemanticSnippetExtractor

            # Get the reranker (will reuse cached instance)
            reranker = get_reranker(reranker_model)

            # Create snippet extractor that reuses the reranker's model
            _snippet_extractor_cache[reranker_model] = SemanticSnippetExtractor(
                reranker=reranker,
                min_paragraph_length=50,
                max_paragraph_length=500,
                max_paragraphs_to_score=20,
            )
            logger.info(f"Snippet extractor initialized for model: {reranker_model}")
        except Exception as e:
            logger.error(f"Failed to initialize snippet extractor: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize snippet extractor: {str(e)}",
            ) from e

    return _snippet_extractor_cache[reranker_model]


def get_query_expander() -> "GeminiQueryExpander":
    """Get or initialize the query expander singleton.

    Returns:
        GeminiQueryExpander instance

    Raises:
        HTTPException: If query expander cannot be initialized
    """
    global _query_expander

    if _query_expander is None:
        try:
            logger.info("Initializing Gemini query expander")
            from indexa.query_expansion.gemini_expander import GeminiQueryExpander

            # Get tokenizer from search engine to ensure query terms match index
            engine = get_search_engine()

            _query_expander = GeminiQueryExpander(
                max_expanded_terms=5,  # Conservative expansion
                tokenizer=engine.tokenizer,  # Use same tokenizer as indexer
            )
            logger.info("Query expander initialized successfully with tokenizer")
        except Exception as e:
            logger.error(f"Failed to initialize query expander: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize query expander: {str(e)}. "
                "Make sure GEMINI_API_KEY is set in environment variables.",
            ) from e

    return _query_expander


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
                use_field_weights=True,  # Enable BM25F field weighting (title, heading, lead, body)
                load_embeddings=True,  # NEW: Preload embeddings on initialization
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
    rerank: bool = Query(False, description="Enable neural reranking (Assignment 2)"),
    num_candidates: int = Query(
        100, description="Number of BM25 candidates for reranking", ge=10, le=500
    ),
    reranker_model: str = Query(
        "unicamp-dl", description="Reranker model: 'unicamp-dl' or 'mmarco'"
    ),
    generate_answers: bool = Query(
        False, description="Generate AI answer using Gemini (Assignment 2)"
    ),
    expand_query: bool = Query(
        False, description="Enable query expansion with Gemini (Assignment 2)"
    ),
    extract_snippets: bool = Query(
        False, description="Enable semantic snippet extraction for top 10 results (Assignment 2)"
    ),
) -> SearchResponse:
    """Search for documents matching the given query using BM25 ranking.

    With neural reranking enabled (Assignment 2), uses a two-stage pipeline:
    1. BM25 retrieves top candidates (fast, lexical)
    2. Cross-encoder reranks candidates (slower, semantic)

    With answer generation enabled (Assignment 2), generates a natural language
    answer using Gemini LLM based on the top-ranked document.

    With query expansion enabled (Assignment 2), automatically adds related terms
    to improve recall while maintaining transparency by showing expanded terms.

    With semantic snippet extraction enabled (Assignment 2), uses the neural reranker
    to find the most relevant paragraph within each of the top 10 documents.

    Args:
        query: The search query string
        num_results: Maximum number of results to return (1-100)
        min_score: Minimum BM25 score threshold (default: 0.0)
        k1: BM25 k1 parameter for term frequency saturation (default: 1.2)
        b: BM25 b parameter for length normalization (default: 0.75)
        rerank: Enable neural reranking (default: False)
        num_candidates: Number of BM25 candidates to retrieve before reranking (default: 100)
        reranker_model: Cross-encoder model to use (default: 'unicamp-dl')
        generate_answers: Enable AI answer generation with Gemini (default: False)
        expand_query: Enable query expansion with Gemini (default: False)
        extract_snippets: Enable semantic snippet extraction (default: False)

    Returns:
        SearchResponse with ranked results, optional AI-generated answers, and expansion info
    """
    start_time = time.time()

    try:
        engine = get_search_engine()

        # Query expansion (Assignment 2 - Task 3)
        original_query = query
        expanded_terms: list[str] = []
        query_expansion_used = False

        if expand_query:
            try:
                logger.info(f"Query expansion enabled for: '{query}'")
                expander = get_query_expander()
                expansion_result = expander.expand_query(query)

                # Check if expansion was skipped by gating rules
                if expansion_result.get("skipped", False):
                    logger.info(
                        f"Query expansion skipped: {expansion_result.get('skip_reason', 'Unknown')}"
                    )
                    # Don't expand - keep original query
                    expanded_terms = []
                    query_expansion_used = False

                elif expansion_result["success"] and expansion_result["expanded_terms"]:
                    # Use WEIGHTED TERMS for search (Rocchio-style relevance feedback)
                    query = expansion_result["weighted_terms"]
                    expanded_terms = expansion_result["expanded_terms"]
                    query_expansion_used = True

                    # Log weighted terms for debugging
                    logger.debug(f"Weighted terms: {query}")
                    logger.info(
                        f"Query expanded: '{original_query}' + {len(expanded_terms)} terms "
                        f"(original_weight=1.0, expanded_weight=0.4)"
                    )
                else:
                    logger.info(f"No expansion needed or failed for query: '{original_query}'")

            except Exception as e:
                # Don't fail the entire search if expansion fails
                logger.error(f"Query expansion failed, proceeding with original query: {e}")
                query = original_query

        # Update BM25 parameters if different from defaults
        if k1 != engine.k1 or b != engine.b:
            engine.bm25_scorer.k1 = k1
            engine.bm25_scorer.b = b
            engine.k1 = k1
            engine.b = b

        # Perform search (with or without reranking)
        if rerank:
            logger.info(f"Neural reranking enabled with model '{reranker_model}'")
            # Use pre-loaded reranker singleton for better performance
            reranker = get_reranker(reranker_model)
            search_results = engine.search_with_reranking(
                query=query,
                num_results=num_results,
                num_candidates=num_candidates,
                min_score=min_score,
                reranker_model=reranker_model,
                reranker=reranker,
            )
        else:
            search_results = engine.search(
                query=query,
                num_results=num_results,
                min_score=min_score,
            )

        # Convert to API Document model
        documents = []
        for result in search_results:
            doc = Document(
                id=result.doc_id,
                title=result.title,
                content=result.content_preview,
                score=result.score,
                url=result.url,
            )
            # Add BM25 and neural scores if reranking was used
            if rerank and hasattr(result, "bm25_score") and hasattr(result, "neural_score"):
                doc.bm25_score = result.bm25_score  # type: ignore
                doc.neural_score = result.neural_score  # type: ignore
            documents.append(doc)

        # Semantic snippet extraction (Assignment 2 - Task 3)
        if extract_snippets and documents:
            try:
                # Only extract snippets for top 10 results (performance optimization)
                num_docs_for_snippets = min(10, len(documents))
                logger.info(f"Extracting semantic snippets for top {num_docs_for_snippets} results")

                # Get snippet extractor (reuses reranker model)
                snippet_extractor = get_snippet_extractor(reranker_model)

                # Determine query for snippet extraction
                # Use original query (not expanded) for better snippet relevance
                snippet_query = original_query if query_expansion_used else query
                if isinstance(snippet_query, list):
                    # If query is weighted terms, extract original terms
                    snippet_query = " ".join(
                        [
                            str(item.get("term", "")) if isinstance(item, dict) else str(item)
                            for item in snippet_query
                        ]
                    )

                # Extract snippets for top N documents
                for i in range(num_docs_for_snippets):
                    doc = documents[i]
                    result = search_results[i]

                    # Get full document content
                    full_doc = engine.get_full_document(result.doc_id)

                    if full_doc and full_doc.get("content"):
                        # Extract semantic snippet
                        snippet = snippet_extractor.extract_snippet(
                            query=snippet_query,
                            document_content=full_doc["content"],
                            max_snippet_length=300,
                        )
                        # Update document content with semantic snippet
                        doc.content = snippet
                        logger.debug(f"Extracted snippet for doc {doc.id}: {snippet[:100]}...")
                    else:
                        logger.warning(
                            f"Full content not available for doc {result.doc_id}, "
                            "keeping default preview"
                        )

                logger.info(f"Snippet extraction completed for {num_docs_for_snippets} documents")

            except Exception as e:
                logger.error(f"Error extracting snippets: {e}", exc_info=True)
                # Don't fail the entire request if snippet extraction fails
                # Just keep the default previews

        answer_snippets = []
        if generate_answers and documents:
            try:
                logger.info("Generating AI answer with Gemini for top 3 documents")
                answer_gen = get_answer_generator()

                # Get full document content for top 3 documents (or fewer if not enough results)
                top_k = min(3, len(search_results))
                docs_for_generation = []

                for i in range(top_k):
                    result = search_results[i]
                    full_doc = engine.get_full_document(result.doc_id)

                    if full_doc:
                        docs_for_generation.append(
                            {"title": full_doc["title"], "text": full_doc["content"]}
                        )
                    else:
                        # Fallback to preview if full content not available
                        logger.warning(
                            f"Full content not available for doc {result.doc_id}, using preview"
                        )
                        docs_for_generation.append(
                            {"title": result.title, "text": result.content_preview}
                        )

                # Generate answer based on top 3 documents
                answer_result = answer_gen.generate_answer(
                    query=query,
                    documents=docs_for_generation,
                )

                # Map source indices (1-based) to actual document IDs
                sources_used = answer_result.get("sources_used", [])
                source_doc_ids = []
                for idx in sources_used:
                    # Convert 1-based index to 0-based and check bounds
                    if 1 <= idx <= top_k:
                        source_doc_ids.append(search_results[idx - 1].doc_id)

                # Build answer snippet with source document IDs
                answer_snippets.append(
                    AnswerSnippet(
                        doc_id=search_results[0].doc_id,
                        answer=answer_result["answer"],
                        success=answer_result["success"],
                        error=answer_result.get("error"),
                        sources_used=source_doc_ids,
                    )
                )

                logger.info(
                    f"Generated answer successfully using {top_k} documents. "
                    f"Sources cited: {sources_used} (doc_ids: {source_doc_ids})"
                )

            except Exception as e:
                logger.error(f"Error generating answer: {e}", exc_info=True)
                # Don't fail the entire request if answer generation fails
                # Just return empty answer_snippets

        execution_time = (time.time() - start_time) * 1000  # Convert to ms

        # Build display query (original + expanded terms for user visibility)
        display_query = original_query
        if query_expansion_used and expanded_terms:
            display_query = f"{original_query} {' '.join(expanded_terms)}"

        logger.info(
            f"Query '{original_query}' returned {len(documents)} results in {execution_time:.2f}ms"
        )

        return SearchResponse(
            results=documents,
            query=display_query,  # Show full expanded query to user
            num_results=len(documents),
            execution_time_ms=round(execution_time, 2),
            answer_snippets=answer_snippets,
            # Query expansion fields
            original_query=original_query if query_expansion_used else None,
            expanded_terms=expanded_terms,
            query_expansion_used=query_expansion_used,
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
            "Similarity algorithm: 'embedding' (semantic, best), 'cosine', 'field-weighted', "
            "'jaccard', 'hybrid', or 'bm25' (default: hybrid)"
        ),
        regex="^(embedding|cosine|field-weighted|jaccard|hybrid|bm25)$",
    ),
) -> SearchResponse:
    """Search for documents similar to the given document (relevance feedback).

    This endpoint implements relevance feedback by finding documents that are
    similar to a specified document. It's useful for "more like this" functionality.

    Supported similarity methods:
    - 'embedding': Semantic similarity using precomputed embeddings (BEST, Assignment 2)
        Requires embeddings generated via: python scripts/generate_embeddings.py
    - 'hybrid': Combines TF-IDF cosine and Jaccard (70/30 weight) (default)
    - 'cosine': TF-IDF cosine similarity (fast, lexical)
    - 'field-weighted': Field-aware cosine similarity (uses title/heading/lead/body structure)
    - 'jaccard': Jaccard coefficient (set-based similarity)
    - 'bm25': Legacy BM25-based method (for backward compatibility)

    Args:
        doc_id: The ID of the document to find similar documents for
        num_results: Maximum number of results to return (1-100)
        min_score: Minimum similarity score threshold (0.0-1.0 for embedding, unbounded for others)
        similarity_method: Similarity algorithm to use (default: 'hybrid')

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


@router.get("/preload_model/{model_name}")
def preload_model(model_name: str = Path(..., description="Model name to preload")) -> dict:
    """Preload a neural reranking model to warm the cache.

    This endpoint allows the frontend to proactively load models when the user
    changes the model selection in the UI, avoiding delays on the first query.

    Supported models:
    - unicamp-dl: Lightweight Portuguese model
    - mmarco: Multilingual model (heavier)

    Args:
        model_name: The name of the model to preload

    Returns:
        Dictionary with model information
    """
    try:
        logger.info(f"Preload request for model: {model_name}")

        # Validate model name
        from indexa.reranker.neural_reranker import SUPPORTED_MODELS

        if model_name not in SUPPORTED_MODELS:
            supported = list(SUPPORTED_MODELS.keys())
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {model_name}. Supported models: {supported}",
            )

        # Load model (uses cache if already loaded)
        reranker = get_reranker(model_name)
        model_info = reranker.get_model_info()

        logger.info(f"Model {model_name} is ready: {model_info}")

        return {
            "status": "ready",
            "model_name": model_name,
            "model_info": model_info,
            "message": f"Model '{model_name}' is loaded and ready",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preload model error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to preload model: {str(e)}") from e


@router.get("/embeddings/status")
def get_embeddings_status() -> dict:
    """Get status of document embeddings for semantic similarity search.

    Returns information about whether embeddings are loaded, their size,
    and doc_id mapping status.

    Returns:
        Dictionary with embedding status and statistics
    """
    try:
        engine = get_search_engine()

        if engine.embedding_similarity is not None and engine._embeddings_available:
            info = engine.embedding_similarity.get_info()
            return {
                "status": "loaded",
                "is_loaded": info["is_loaded"],
                "num_documents": info["num_documents"],
                "embedding_dim": info["embedding_dim"],
                "memory_mb": round(info["memory_usage_mb"], 2),
                "has_mapping": info["has_mapping"],
                "doc_id_range": info.get("doc_id_range"),
                "message": "Embeddings are loaded and ready for similarity search",
            }
        else:
            return {
                "status": "not_loaded",
                "is_loaded": False,
                "message": (
                    "Embeddings not loaded. They may not be available or need to be generated."
                ),
            }

    except Exception as e:
        logger.error(f"Error checking embedding status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to check embedding status: {str(e)}"
        ) from e


@router.get("/document/{doc_id}/paragraph_scores")
def get_document_paragraph_scores(
    doc_id: int = Path(..., description="Document ID", ge=0),
    query: str = Query(..., description="Query to score paragraphs against", min_length=1),
    reranker_model: str = Query(
        "unicamp-dl", description="Reranker model: 'unicamp-dl' or 'mmarco'"
    ),
) -> dict:
    """Get paragraph-level relevance scores for a document.

    This endpoint scores all paragraphs in a document against a given query
    using the neural reranker, allowing users to see why specific content was
    considered relevant.

    Args:
        doc_id: The document ID
        query: Query to score paragraphs against
        reranker_model: Cross-encoder model to use (default: 'unicamp-dl')

    Returns:
        Dictionary with paragraph scores and metadata
    """
    try:
        engine = get_search_engine()

        # Get full document content
        full_doc = engine.get_full_document(doc_id)

        if not full_doc:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID {doc_id} not found",
            )

        # Get snippet extractor (reuses reranker model)
        snippet_extractor = get_snippet_extractor(reranker_model)

        # Extract snippet with scores
        result = snippet_extractor.extract_snippet_with_scores(
            query=query,
            document_content=full_doc["content"],
            max_snippet_length=300,
        )

        # Return paragraph scores with document metadata
        return {
            "doc_id": doc_id,
            "title": full_doc["title"],
            "query": query,
            "snippet": result["snippet"],
            "best_score": result["best_score"],
            "paragraph_scores": result["paragraph_scores"],
            "num_paragraphs": len(result["paragraph_scores"]),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get paragraph scores error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get paragraph scores: {str(e)}"
        ) from e


@router.post("/preload_embeddings")
def preload_embeddings() -> dict:
    """Preload document embeddings for semantic similarity search.

    This endpoint triggers loading of embeddings if they haven't been loaded yet.
    Embeddings are large (~3GB+) and take several seconds to load, so this
    can be called proactively to warm up the cache.

    Returns:
        Dictionary with preload status and embedding information
    """
    try:
        logger.info("Preload request for embeddings")
        engine = get_search_engine()

        # Trigger embedding load if not already loaded
        if not engine._embeddings_available:
            logger.info("Embeddings not loaded, triggering load...")
            engine._load_embeddings()

        # Check status after load attempt
        if engine.embedding_similarity is not None and engine._embeddings_available:
            info = engine.embedding_similarity.get_info()
            logger.info(f"Embeddings preloaded successfully: {info}")

            return {
                "status": "ready",
                "is_loaded": True,
                "num_documents": info["num_documents"],
                "embedding_dim": info["embedding_dim"],
                "memory_mb": round(info["memory_usage_mb"], 2),
                "has_mapping": info["has_mapping"],
                "doc_id_range": info.get("doc_id_range"),
                "message": "Embeddings are loaded and ready for similarity search",
            }
        else:
            return {
                "status": "not_available",
                "is_loaded": False,
                "message": (
                    "Embeddings could not be loaded. They may not exist. "
                    "Generate embeddings using: python scripts/generate_embeddings.py"
                ),
            }

    except Exception as e:
        logger.error(f"Preload embeddings error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to preload embeddings: {str(e)}"
        ) from e
