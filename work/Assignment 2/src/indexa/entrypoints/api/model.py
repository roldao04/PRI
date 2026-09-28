from pydantic import BaseModel

from indexa.core.model import Document


class SearchRequest(BaseModel):
    query: str
    num_results: int = 10
    min_score: float = 0.0


class AnswerSnippet(BaseModel):
    """Generated answer snippet from LLM based on a document."""

    doc_id: int
    answer: str
    success: bool = True
    error: str | None = None
    sources_used: list[int] = []  # List of document IDs that were actually used in the answer


class SearchResponse(BaseModel):
    results: list[Document]
    query: str = ""
    num_results: int = 0
    execution_time_ms: float = 0.0
    answer_snippets: list[AnswerSnippet] = []
    # Query expansion fields (Assignment 2 - Task 3)
    original_query: str | None = None
    expanded_terms: list[str] = []
    query_expansion_used: bool = False


class SimilarSearchRequest(BaseModel):
    doc_id: int
    num_results: int = 10
    min_score: float = 0.0


class IndexStatsResponse(BaseModel):
    num_documents: int
    num_terms: int
    avg_doc_length: float
    bm25_k1: float
    bm25_b: float
