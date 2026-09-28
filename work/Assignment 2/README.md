# Assignment 2 - Semantic Search System with AI Enhancements

Building a modern semantic search system by integrating neural networks on top of the classic BM25 retrieval from Assignment 1.

## Overview

This project extends Assignment 1's information retrieval system with AI capabilities:

- **Neural Reranking**: Improve ranking quality using cross-encoder models
- **Answer Generation (RAG)**: Provide natural language answers using LLMs
- **Advanced Semantic Features**: Semantic snippets, embedding-based similarity, query expansion

## Setup & Download

### Prerequisites
- Python 3.13+
- Pre-built index: [Download wiki_index_full.zip (7.5GB)](https://drive.google.com/file/d/13ssCSkmYvj1x6-YPcNhR3C25IoqxPevh/view?usp=sharing)
- Google Gemini API key (for answer generation & query expansion)

### Quick Start
```bash
# 1. Install dependencies
uv sync

# 2. Download and extract index (if not present)
# Extract wiki_index_full.zip to project root

# 3. Configure API key
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 4. Start the server
uvicorn indexa.entrypoints.asgi:app --reload

# 5. Open browser
# http://localhost:8000/static_pages/index.html
```

## Project Structure

```
Assignment 2/
├── src/indexa/           # Main codebase
│   ├── core/             # Logging, models
│   ├── entrypoints/      # CLI and API
│   ├── indexer/          # SPIMI, tokenizer
│   ├── searcher/         # BM25, search engine
│   ├── reranker/         # Neural reranking
│   ├── answer_generator/ # RAG with Gemini
│   ├── query_expansion/  # Gemini Query Expansion
│   ├── snippet/          # Semantic snippet extraction
│   └── embeddings/       # Document embeddings
├── wiki_index_full/      # Pre-built index (download separately)
├── static_pages/         # Web interface
├── docs/                 # Documentation
├── pyproject.toml        # Dependencies
├── README.md             # This file
```

## Implementation Details & Design Decisions

This section addresses the assignment submission requirements, detailing our implementation choices and rationale for all AI components.

### Neural Reranking

**Implementation:** Two-stage retrieval pipeline using cross-encoder models for semantic reranking.

**Model Choice & Rationale:**

**Primary:** `unicamp-dl/mMiniLM-L6-v2-pt-v2` (Portuguese-optimized)
- **Why Portuguese-specific?** Our corpus is 1.15M Portuguese Wikipedia articles. A model trained on Portuguese text (PT-BR/PT-PT) understands linguistic nuances (conjugations, morphology, regional vocabulary) that multilingual models miss. This directly improves semantic matching quality.
- **Size tradeoff:** 6 layers (~90MB) vs 12 layers (~120MB) for mmarco. We prioritized faster inference (700-900ms vs 1200-1500ms on GPU) while maintaining quality, as reranking happens on every query.

**Alternative:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Multilingual)
- Explored for comparison. Supports 100+ languages but slower and less optimized for Portuguese. Available via `reranker_model=mmarco` parameter for users who need multilingual support.

**Architecture Decision: Why Two-Stage?**
- **Pure neural search** (1.15M documents) is too slow (~30s+ per query)
- **Pure BM25** misses semantic relationships (e.g., "capital Portugal" → "Lisboa")
- **Hybrid approach:** BM25 retrieves top 100 candidates in 50-200ms (fast, lexical recall), then cross-encoder reranks in 700-900ms (slow, semantic precision). Total: <1.5s for high-quality results.

**Usage:** `GET /search?query=história&rerank=true&reranker_model=unicamp-dl`

*See [docs/NEURAL_RERANKING.md](docs/NEURAL_RERANKING.md) for performance benchmarks and detailed analysis.*

---

### Answer Generation

**Information Fed to LLM:**

We send **exactly 3 documents** (title + full content) to Gemini after BM25/reranking:
- **Why 3 docs?** Balances context richness (multiple sources for cross-validation) vs cost/latency. Single doc risks incomplete information; 5+ docs increase token costs and dilute focus.
- **Why full content?** No truncation ensures LLM has complete context. Our Wikipedia articles average 500-1500 words—within Gemini's context window. Truncation risks losing relevant information.
- **Structured prompt:** Portuguese (pt-PT) instructions enforce conciseness (2-3 sentences), source attribution (cite document numbers), and fact-grounding (prohibit hallucination).

**Model:** `gemini-2.5-flash-lite`
- **Why this model?** Optimized for speed (200-500ms response) and generous free tier quota (1500 requests/day). Gemini Pro is slower and costs money; Flash maintains quality for factual Q&A while being production-ready.

**Prompt Design:** Structured in 3 sections:
1. **System instructions:** "Motor de busca inteligente" role, pt-PT language, conciseness requirement
2. **Context:** 3 documents with titles + full text
3. **Output format:** Answer + source citations [FONTES: 1, 2, 3]

**Usage:** `GET /search?query=capital%20de%20portugal&generate_answers=true`

*See [docs/ANSWER_GENERATION.md](docs/ANSWER_GENERATION.md) for complete prompt structure.*

---

### Additional AI Enhancements

#### **A. Semantic Snippet Extraction**

**Design Decision:** Reuse the same cross-encoder model from neural reranking.
- **Why reuse?** Avoids loading a second neural model (saves ~500MB memory + 2-3s startup). The cross-encoder already scores query-document relevance—applying it to paragraphs is identical scoring logic.
- **Implementation:** Split documents into paragraphs (50-500 chars), score top 20 paragraphs per document, select highest-scoring 300-char snippet.
- **Benefit:** Users see **why** a document is relevant (e.g., for "batalha de Aljubarrota", shows the battle paragraph instead of generic intro).

**Usage:** `GET /search?query=história&extract_snippets=true`

#### **B. Embedding-Based Relevance Feedback**

**Design Decision:** Precompute 384-dim embeddings for all 1.15M documents offline.
- **Why precompute?** Real-time embedding generation takes ~5-10min for 1M docs. Precomputation trades disk space (3.5GB) for instant similarity search (<50ms).
- **Why 384 dimensions?** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` output size. Balances semantic richness vs memory efficiency (768-dim models use 2x space).
- **Benefit:** Enables semantic "More Like This"—finds documents with similar **meaning** rather than just lexical overlap (e.g., "futebol" → "desporto", "jogador").

**Storage:** Embeddings included in `wiki_index_full.zip` (embeddings.npy + doc_id_mapping.npy)

**Usage:** `GET /search_similar?doc_id=123&similarity_method=embedding`

*See [docs/EMBEDDINGS.md](docs/EMBEDDINGS.md) for generation process.*

#### **C. Query Expansion with LLM**

**Design Decision:** Conservative weighting (original: 1.0, expanded: 0.4) with intelligent gating.
- **Why conservative?** Aggressive expansion (0.7-1.0 weights) degrades precision—expanded terms can drift from user intent. We prioritize exact matches (title, lead) via BM25F field weights (3x-5x multipliers), so expanded terms must stay secondary.
- **Why gating rules?** Not all queries benefit from expansion:
  - **Skip 1-word queries:** Already specific (e.g., "Lisboa")
  - **Skip 6+ word queries:** Already detailed (e.g., "história da batalha de Aljubarrota em 1385")
  - **Expand 2-5 word queries:** Sweet spot (e.g., "animais marinhos" → add "oceano", "mar", "fauna aquática")
- **Gemini decision-making:** For edge cases (2-5 words), LLM evaluates whether query is specific (name, place) or generic (topic, concept).

**Benefit:** Improves recall for vocabulary mismatch (e.g., "IA" → "inteligência artificial", "machine learning") without hurting precision.

**Usage:** `GET /search?query=animais&expand_query=true`

*See [docs/QUERY_EXPANSION.md](docs/QUERY_EXPANSION.md) for weighting rationale and gating logic.*

## Authors

- André Alves (aaalves@ua.pt)
- João Roldão (jroldao04@ua.pt)

## License

University of Aveiro - PRI 2025 - Assignment 2
