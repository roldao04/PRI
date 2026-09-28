from pydantic import BaseModel

from indexa.core.model import Document


class SearchRequest(BaseModel):
    query: str
    num_results: int = 10
    min_score: float = 0.0


class SearchResponse(BaseModel):
    results: list[Document]
    query: str = ""
    num_results: int = 0
    execution_time_ms: float = 0.0


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
