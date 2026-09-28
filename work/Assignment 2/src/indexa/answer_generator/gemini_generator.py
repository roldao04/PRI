"""Gemini-based answer generator for RAG (Retrieval-Augmented Generation)."""

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import to avoid dependency issues
_genai = None


def _get_genai():
    """Lazy load google.generativeai to avoid import errors when API key is missing."""
    global _genai
    if _genai is None:
        try:
            import google.generativeai as genai

            _genai = genai
        except ImportError as e:
            raise ImportError("google-generativeai package not found. ") from e
    return _genai


class GeminiAnswerGenerator:
    """Generates natural language answers using Google Gemini API.

    This class implements the Answer Generation component (Task 2) from Assignment 2.
    It uses Gemini to generate concise answers based on retrieved documents.
    """

    def __init__(self, api_key: str | None = None, model_name: str = "gemini-2.5-flash-lite"):
        """Initialize Gemini answer generator.

        Args:
            api_key: Google API key for Gemini. If None, reads from GEMINI_API_KEY env var
            model_name: Gemini model to use(default:gemini-2.5-flash-lite for better quota support)

        Raises:
            ValueError: If API key is not provided or found in environment
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. "
                "Set it as environment variable or pass it to constructor."
            )

        self.model_name = model_name

        # Configure Gemini
        genai = _get_genai()
        genai.configure(api_key=self.api_key)  # type: ignore

        # Initialize model
        self.model = genai.GenerativeModel(model_name)  # type: ignore

        logger.info(f"Initialized Gemini answer generator with model: {model_name}")

    def _build_prompt(self, query: str, documents: list[dict[str, str]]) -> str:
        """Build the prompt for Gemini to generate an answer.

        Args:
            query: User's search query
            documents: List of documents, each with 'title' and 'text' keys

        Returns:
            Formatted prompt string
        """
        # Build documents section
        docs_text = ""
        for i, doc in enumerate(documents, 1):
            docs_text += f"""**Documento {i}:**
Título: {doc['title']}
Conteúdo: {doc['text']}

"""

        prompt = f"""És um motor de busca inteligente. A tua tarefa é responder \
à pergunta do utilizador de forma concisa e direta com base APENAS nos documentos \
fornecidos.

**Instruções:**
- Responde em português de Portugal (pt-PT)
- Sê conciso (máximo 2-3 frases)
- Responde apenas com conteúdo relevante ao que foi perguntado
- Usa informações APENAS dos documentos fornecidos
- Se os documentos não contêm a resposta, diz "Não há informação suficiente \
nos documentos para responder."
- Não inventes informações
- Podes combinar informação de múltiplos documentos se relevante
- IMPORTANTE: Depois da resposta, numa nova linha, indica quais documentos usaste \
no formato: [FONTES: 1, 2, 3] (usando os números dos documentos que realmente usaste)

**Pergunta do utilizador:**
{query}

{docs_text}**Resposta:**"""
        return prompt

    def _parse_sources(self, text: str) -> tuple[str, list[int]]:
        """Parse the response to extract answer and source document indices.

        Args:
            text: Raw response from Gemini

        Returns:
            Tuple of (answer_text, list of document indices used)
        """
        # Look for [FONTES: 1, 2, 3] pattern
        pattern = r"\[FONTES:\s*([\d,\s]+)\]"
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            # Extract answer (everything before [FONTES:])
            answer = text[: match.start()].strip()

            # Extract document indices
            sources_str = match.group(1)
            source_indices = [int(s.strip()) for s in sources_str.split(",") if s.strip().isdigit()]

            return answer, source_indices

        # If no sources found, return full text as answer with empty sources
        return text.strip(), []

    def generate_answer(self, query: str, documents: list[dict[str, str]]) -> dict[str, Any]:
        """Generate answer using Gemini based on multiple documents.

        Args:
            query: User's search query
            documents: List of documents, each with 'title' and 'text' keys

        Returns:
            Dictionary with 'answer', 'success', 'error', and 'sources_used' keys
        """
        try:
            prompt = self._build_prompt(query, documents)
            response = self.model.generate_content(prompt)
            raw_answer = response.text.strip()

            # Parse answer and extract sources
            answer, source_indices = self._parse_sources(raw_answer)

            logger.info(
                f"Generated answer for query '{query}' using {len(documents)} documents. "
                f"Sources cited: {source_indices}. Answer: {answer[:100]}..."
            )

            return {
                "answer": answer,
                "success": True,
                "error": None,
                "sources_used": source_indices,  # List of document indices (1-based)
            }

        except Exception as e:
            logger.error(f"Error generating answer with Gemini: {e}", exc_info=True)
            return {
                "answer": "Erro ao gerar resposta.",
                "success": False,
                "error": str(e),
                "sources_used": [],
            }
