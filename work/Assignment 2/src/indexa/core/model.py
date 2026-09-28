from pydantic import BaseModel


class Document(BaseModel):
    """Represents a single document in the search results."""

    id: int
    title: str
    content: str
    score: float = 0.0
    url: str = ""
    bm25_score: float | None = None
    neural_score: float | None = None
