"""Gemini-based query expansion for improving search recall.

This module implements query expansion using Google Gemini API to automatically
add related terms, synonyms, and alternate phrasings to user queries, helping
bridge the vocabulary gap between queries and relevant documents.
"""

import logging
import os
import re
from typing import Any, cast

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
            raise ImportError(
                "google-generativeai package not found. "
                "Install it with: pip install google-generativeai"
            ) from e
    return _genai


class GeminiQueryExpander:
    """Expands search queries using Google Gemini API.

    This class implements query expansion to improve search recall by:
    - Adding synonyms and related terms
    - Including alternate vocabulary and phrasings
    - Handling Portuguese-specific terminology

    The expansion is conservative to minimize precision loss while improving
    recall for queries with vocabulary mismatch.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.0-flash",
        max_expanded_terms: int = 5,
        original_term_weight: float = 1.0,
        expanded_term_weight: float = 0.4,
        tokenizer=None,
    ):
        """Initialize Gemini query expander.

        Args:
            api_key: Google API key for Gemini. If None, reads from GEMINI_API_KEY env var
            model_name: Gemini model to use (default: gemini-2.0-flash for speed)
            max_expanded_terms: Maximum number of terms to add (default: 5, conservative)
            original_term_weight: Weight for original query terms (default: 1.0)
            expanded_term_weight: Weight for expanded terms (default: 0.4, following Rocchio)
            tokenizer: Tokenizer instance for normalizing query terms
                (required for proper index matching)

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
        self.max_expanded_terms = max_expanded_terms
        self.original_term_weight = original_term_weight
        self.expanded_term_weight = expanded_term_weight
        self.tokenizer = tokenizer

        # Warn if tokenizer is not provided (will fall back to simple split)
        if self.tokenizer is None:
            logger.warning(
                "No tokenizer provided to GeminiQueryExpander. "
                "Query terms will not be stemmed/normalized, which may cause index mismatches. "
                "Pass tokenizer=engine.tokenizer to fix this."
            )

        # Configure Gemini
        genai = _get_genai()
        genai.configure(api_key=self.api_key)  # type: ignore

        # Initialize model with low temperature for consistency
        generation_config = {
            "temperature": 0.3,  # Low temperature for more deterministic output
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 200,  # Short output (just a list of terms)
        }

        self.model = genai.GenerativeModel(  # type: ignore
            model_name=model_name, generation_config=cast(Any, generation_config)
        )

        logger.info(
            f"Initialized Gemini query expander with model: {model_name}, "
            f"max_terms: {max_expanded_terms}"
        )

    def _is_question(self, query: str) -> bool:
        """Detect if query is exploratory/question-based.

        Args:
            query: User query string

        Returns:
            True if query appears to be a question
        """
        question_words = [
            "como",
            "o que",
            "qual",
            "quais",
            "quando",
            "onde",
            "porquê",
            "porque",
            "quem",
            "quanto",
        ]
        query_lower = query.lower()

        # Check for question words or question mark
        return any(word in query_lower for word in question_words) or query.strip().endswith("?")

    def should_expand_query(self, query: str) -> tuple[bool, str]:
        """Determine if query should be expanded using hybrid gating.

        Hybrid approach:
        - Hard rules for obvious cases (1 word, 6+ non-questions)
        - Gemini decides for edge cases (2-5 words)

        Args:
            query: User query string

        Returns:
            (should_expand: bool, reason: str)
        """
        query_terms = query.split()
        num_words = len(query_terms)

        # HARD RULE 1: Never expand 1-word queries (too specific)
        if num_words == 1:
            logger.info(f"Query expansion SKIPPED: 1 word query '{query}' is likely specific")
            return False, "Single-word query (likely already specific)"

        # HARD RULE 2: Skip detailed queries (6+ words) unless it's a question
        if num_words >= 6:
            if self._is_question(query):
                logger.info("Query expansion ENABLED: 6+ word question detected")
                return True, "Long query but appears to be a question (exploratory)"
            else:
                logger.info(f"Query expansion SKIPPED: {num_words} words (already detailed)")
                return False, f"Long query ({num_words} words) already detailed"

        # SOFT RULE: Let Gemini decide for 2-5 words (sweet spot, needs context)
        logger.info(f"Query expansion: Gemini will decide for {num_words}-word query")
        return True, f"Medium length query ({num_words} words) - Gemini will evaluate"

    def _build_expansion_prompt(self, query: str, num_words: int | None = None) -> str:
        """Build the prompt for Gemini to expand the query.

        Args:
            query: User's original search query
            num_words: Number of words in query (for context)

        Returns:
            Formatted prompt string for conservative query expansion
        """
        # Calculate num_words if not provided
        if num_words is None:
            num_words = len(query.split())

        # Determine instruction based on word count
        if num_words == 2:
            word_count_instruction = """
**ATENÇÃO:** Esta consulta tem apenas 2 palavras.
- Se for um nome próprio ou lugar específico → "NENHUM"
- Se for uma combinação específica (ex: "Steve Jobs", "Lisboa Portugal") → "NENHUM"
- Se for genérica/vaga (ex: "animais marinhos", "aquecimento global") → Expande normalmente
"""
        elif 3 <= num_words <= 5:
            word_count_instruction = """
**ATENÇÃO:** Esta consulta tem 3-5 palavras (zona ideal para expansão).
- Expande com sinónimos e termos relacionados
- Mantém o foco no contexto da consulta
"""
        else:
            word_count_instruction = ""

        prompt = f"""És um assistente especializado em expandir consultas de pesquisa em português.

**Consulta do utilizador:** {query}
**Número de palavras:** {num_words}

{word_count_instruction}

**Tarefa:** Expandir a consulta com até {self.max_expanded_terms} termos relacionados relevantes.

**Instruções:**
- Adiciona APENAS sinónimos diretos e termos fortemente relacionados
- Mantém o mesmo nível de especificidade (não generalizes demais)
- Foca em variações vocabulares que ajudem a encontrar documentos relevantes
- NÃO adicionar termos muito genéricos ou vagamente relacionados
- Responde APENAS com os termos adicionais, separados por vírgula
- Se a consulta for muito específica (nomes próprios, lugares, etc.), retorna "NENHUM"
- Se a consulta já tem vocabulário abrangente, retorna "NENHUM"

**Exemplos:**
Consulta: "inteligência artificial" (2 palavras, genérica)
Resposta: aprendizado de máquina, redes neurais, IA, machine learning, deep learning

Consulta: "aquecimento global causas" (3 palavras)
Resposta: mudanças climáticas, crise climática, efeito estufa, alterações climáticas

Consulta: "Lisboa Portugal" (2 palavras, específica)
Resposta: NENHUM

Consulta: "computador" (1 palavra)
Resposta: NENHUM

**Consulta do utilizador:**
{query}

**Termos adicionais (máximo {self.max_expanded_terms}, separados por vírgula):**"""
        return prompt

    def expand_query(self, query: str) -> dict[str, Any]:
        """Expand query with related terms using Gemini.

        Args:
            query: User's original search query

        Returns:
            Dictionary with:
            - 'original_query': Original query string
            - 'expanded_terms': List of additional terms (may be empty)
            - 'expanded_query': Combined query with all terms
            - 'success': Whether expansion succeeded
            - 'skipped': Whether expansion was skipped by gating rules
            - 'skip_reason': Reason for skipping (if skipped)
            - 'error': Error message if failed (None if successful)
        """
        try:
            # STEP 1: Check if query should be expanded (hybrid gating)
            should_expand, reason = self.should_expand_query(query)

            if not should_expand:
                # Hard rule says skip - don't call Gemini
                logger.info(f"Query expansion SKIPPED: {reason}")
                # Build weighted terms with only original query (no expanded terms)
                weighted_terms = self._build_weighted_terms(query, [])
                return {
                    "original_query": query,
                    "expanded_terms": [],
                    "expanded_query": query,
                    "weighted_terms": weighted_terms,
                    "success": True,
                    "skipped": True,
                    "skip_reason": reason,
                    "error": None,
                }

            # STEP 2: Build prompt with context (Gemini will decide for 2-5 words)
            num_words = len(query.split())
            prompt = self._build_expansion_prompt(query, num_words)

            # STEP 3: Call Gemini
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()

            # STEP 4: Parse response
            expanded_terms = self._parse_expanded_terms(response_text)

            # Build expanded query (original + expanded terms)
            expanded_query = f"{query} {' '.join(expanded_terms)}" if expanded_terms else query

            # Build weighted terms structure for BM25 (Rocchio-style weighting)
            weighted_terms = self._build_weighted_terms(query, expanded_terms)

            logger.info(
                f"Query expansion: '{query}' ({num_words} words) -> "
                f"added {len(expanded_terms)} terms: {expanded_terms} "
                f"(weights: original={self.original_term_weight}, "
                f"expanded={self.expanded_term_weight})"
            )

            return {
                "original_query": query,
                "expanded_terms": expanded_terms,
                "expanded_query": expanded_query,
                "weighted_terms": weighted_terms,
                "success": True,
                "skipped": False,
                "skip_reason": None,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Error expanding query '{query}' with Gemini: {e}", exc_info=True)
            # Fallback: return original query without expansion
            weighted_terms = self._build_weighted_terms(query, [])
            return {
                "original_query": query,
                "expanded_terms": [],
                "expanded_query": query,
                "weighted_terms": weighted_terms,
                "success": False,
                "skipped": False,
                "skip_reason": None,
                "error": str(e),
            }

    def _parse_expanded_terms(self, response_text: str) -> list[str]:
        """Parse expanded terms from Gemini response.

        Args:
            response_text: Raw response from Gemini

        Returns:
            List of expanded terms (cleaned and deduplicated)
        """
        # Handle "NENHUM" case (no expansion needed)
        if "nenhum" in response_text.lower():
            return []

        # Split by comma and clean terms
        terms = []
        raw_terms = response_text.split(",")

        for term in raw_terms:
            # Clean: remove quotes, extra whitespace, punctuation
            cleaned = term.strip().strip('"').strip("'").strip()
            cleaned = re.sub(r"[.!?;:]", "", cleaned)  # Remove punctuation

            # Skip empty or very short terms
            if len(cleaned) >= 2:
                terms.append(cleaned.lower())  # Lowercase for consistency

        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        # Limit to max_expanded_terms
        return unique_terms[: self.max_expanded_terms]

    def _build_weighted_terms(self, query: str, expanded_terms: list[str]) -> list[dict[str, Any]]:
        """Build weighted terms structure from original query and expanded terms.

        Implements dynamic weight scaling to ensure expanded terms don't overwhelm
        original terms, maintaining the constraint:
            sum(expanded_weights) ≤ 0.5 × sum(original_weights)

        This ensures that exact matches (especially title matches with BM25F field weights)
        always dominate over documents that only match expanded terms.

        Args:
            query: Original query string
            expanded_terms: List of expanded terms

        Returns:
            List of {"term": str, "weight": float} dicts for BM25 scoring
        """
        weighted_terms = []

        # Tokenize original query using the same tokenizer as the indexer
        # This ensures terms match exactly what's in the index (stemmed, normalized, etc.)
        if self.tokenizer is not None:
            # Use proper tokenization (stemming, normalization, stopword removal, etc.)
            original_terms = [term for term, _ in self.tokenizer.tokenize_with_positions(query)]
        else:
            # Fallback: simple whitespace split (will likely cause index mismatches!)
            original_terms = query.lower().split()
            logger.warning(
                f"Using simple tokenization for query '{query}'. "
                "This may not match index terms (stemming, normalization missing)."
            )

        num_original = len(original_terms)

        # Add original terms with full weight
        for term in original_terms:
            if term:  # Skip empty strings
                weighted_terms.append({"term": term, "weight": self.original_term_weight})

        # Calculate effective expanded weight using dynamic scaling
        # Constraint: sum(expanded_weights) ≤ 0.5 × sum(original_weights)
        # Formula: effective_weight = min(configured_weight, (0.5 × N_orig × W_orig) / N_exp)
        if expanded_terms and num_original > 0:
            # Maximum allowed total weight for expanded terms
            max_total_expanded_weight = 0.5 * num_original * self.original_term_weight
            # Weight per expanded term to stay within constraint
            max_expanded_weight = max_total_expanded_weight / len(expanded_terms)
            # Use the smaller of configured weight or constraint-based weight
            effective_expanded_weight = min(self.expanded_term_weight, max_expanded_weight)

            logger.debug(
                f"Dynamic weight scaling: {num_original} original terms, "
                f"{len(expanded_terms)} expanded terms → "
                f"effective_expanded_weight={effective_expanded_weight:.3f} "
                f"(configured={self.expanded_term_weight}, "
                f"constraint_max={max_expanded_weight:.3f})"
            )
        else:
            effective_expanded_weight = self.expanded_term_weight

        # Add expanded terms with dynamically scaled weight
        # IMPORTANT: Also tokenize expanded terms to match index
        for expanded_term in expanded_terms:
            if not expanded_term:
                continue

            # Tokenize expanded term (may produce multiple tokens or none if stopword)
            if self.tokenizer is not None:
                tokenized_expanded = [
                    t for t, _ in self.tokenizer.tokenize_with_positions(expanded_term)
                ]
                # Add each token from the expanded term
                for token in tokenized_expanded:
                    if token:  # Skip empty
                        weighted_terms.append({"term": token, "weight": effective_expanded_weight})
            else:
                # Fallback: use raw expanded term
                weighted_terms.append(
                    {"term": expanded_term.lower(), "weight": effective_expanded_weight}
                )

        return weighted_terms

    def get_expander_info(self) -> dict[str, Any]:
        """Get information about the query expander configuration.

        Returns:
            Dictionary with expander configuration details
        """
        return {
            "model_name": self.model_name,
            "max_expanded_terms": self.max_expanded_terms,
            "original_term_weight": self.original_term_weight,
            "expanded_term_weight": self.expanded_term_weight,
            "api_configured": bool(self.api_key),
        }
