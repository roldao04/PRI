# Neural Reranking Documentation

## Overview

Neural reranking improves the ranking quality of BM25 search results by using transformer-based cross-encoder models. This adds semantic understanding that goes beyond lexical matching. Indexa implements an optimized two-stage retrieval pipeline with model warming and intelligent candidate selection for production-ready performance.

## How It Works

### Two-Stage Retrieval Pipeline

```
User Query → BM25 Retrieval (fast, top 100) → Neural Reranking (semantic) → Top 10 Results
```

1. **Stage 1 (BM25)**: Fast lexical retrieval to get top candidates (default: 100 documents)
2. **Stage 2 (Neural)**: Semantic reranking using cross-encoder model

**Why Cross-Encoder?**
Cross-encoders process the query and document together, capturing their interaction for better ranking quality. They're slower than bi-encoders but provide superior results for reranking a small set of candidates.

**Performance Optimizations**:
- **Model Warming**: The neural reranker is pre-loaded at startup and kept in memory, eliminating model loading overhead on every query
- **Smart Candidate Selection**: Automatically skips reranking when result sets are too small to benefit (≤10 documents)
- **Dynamic Adjustment**: Adapts the number of candidates to rerank based on available results

## Supported Models

### 1. Unicamp-DL (Default, Recommended)

**Model**: `unicamp-dl/mMiniLM-L6-v2-pt-v2`

- Portuguese-specific cross-encoder
- 6 transformer layers (~90MB)
- Optimized for Portuguese text
- Faster inference
- **Automatically optimized batch size**: GPU=32, CPU=8

### 2. mMarco (Alternative)

**Model**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

- Multilingual cross-encoder (100+ languages)
- 12 transformer layers (~120MB)
- Slower but supports multiple languages
- **Automatically optimized batch size**: GPU=32, CPU=8

## Usage

### API Usage

```bash
# Enable neural reranking
curl "http://localhost:8000/api/v1/search?query=capital%20de%20portugal&rerank=true"

# Use different model
curl "http://localhost:8000/api/v1/search?query=portugal&rerank=true&reranker_model=mmarco"

# Custom number of candidates
curl "http://localhost:8000/api/v1/search?query=portugal&rerank=true&num_candidates=50"
```

**Parameters**:
- `rerank`: Enable neural reranking (default: false)
- `reranker_model`: Model to use - "unicamp-dl" or "mmarco" (default: "unicamp-dl")
- `num_candidates`: Number of BM25 candidates to rerank (default: 100)

**Response Format**:

```json
{
  "results": [
    {
      "id": 42,
      "title": "Lisboa",
      "content": "Lisboa é a capital de Portugal...",
      "score": 12.45,
      "bm25_score": 8.32,
      "neural_score": 12.45,
      "url": "https://pt.wikipedia.org/wiki/Lisboa"
    }
  ],
  "num_results": 10,
  "query": "capital de portugal",
  "execution_time_ms": 823.45
}
```

### Web Interface

1. Click the **⚙️ settings button** to open advanced options
2. Check **"Ativar Reranking Neural (Semântico)"**
3. Select model (Unicamp-DL or mMarco)
4. Search - results show both BM25 and neural scores

### Python Usage

```python
from indexa.searcher import SearchEngine
from pathlib import Path

# Initialize search engine (uses persistent index caching for fast startup)
engine = SearchEngine(index_dir=Path("wiki_index_full"), lazy_loading=True)

# Search with reranking (model is pre-loaded and reused automatically)
results = engine.search_with_reranking(
    query="capital de portugal",
    num_results=10,
    num_candidates=100,
    reranker_model="unicamp-dl"
)

# Results include both scores
for result in results:
    print(f"{result.title}: neural={result.neural_score:.2f}, bm25={result.bm25_score:.2f}")
```

**Using Pre-loaded Reranker** (advanced):
```python
from indexa.reranker import NeuralReranker

# Pre-load reranker once
reranker = NeuralReranker(model_name="unicamp-dl")  # Auto-selects optimal batch size

# Reuse for multiple searches
results = engine.search_with_reranking(
    query="capital de portugal",
    num_results=10,
    reranker=reranker  # Pass pre-loaded instance
)
```

## Implementation Details

### File Structure

```
src/indexa/reranker/
├── __init__.py
└── neural_reranker.py
```

### Key Classes

**`NeuralReranker`** (src/indexa/reranker/neural_reranker.py:22):
- Loads cross-encoder models from HuggingFace
- **Intelligent batch sizing**: Auto-selects GPU=32, CPU=8 for optimal performance
- Auto-detects GPU/CPU
- **Optimized truncation**: Truncates content to ~200 words for faster inference while preserving quality
- Model warming: Can be pre-loaded and reused across queries

**`SearchEngine.search_with_reranking()`** (src/indexa/searcher/search_engine.py:568):
- Orchestrates two-stage pipeline
- Retrieves candidates with BM25
- **Smart reranking**: Automatically skips reranking for small result sets (≤10 documents)
- Accepts pre-loaded reranker instance for performance
- Falls back to BM25 if reranking fails

**`LazyInvertedIndex`** (src/indexa/searcher/index_loader.py:234):
- **Persistent offset caching**: Builds index offset cache on first run, reuses on subsequent runs
- Dramatically reduces startup time from 30-60s to 1-2s after initial build

### Dependencies

```toml
torch>=2.1.0
transformers>=4.35.0
sentence-transformers>=2.2.0
```

### GPU vs CPU

The reranker automatically detects and uses GPU if available, with optimized batch sizes for each:

```python
reranker = NeuralReranker(model_name="unicamp-dl")  # Auto-detects GPU/CPU
# GPU: batch_size=32 (maximizes throughput)
# CPU: batch_size=8 (optimized for memory efficiency)
```

**Performance**:
- CPU: ~1.2s for 100 candidates (optimized)
- GPU: ~0.7s for 100 candidates (3-5x faster)
- First query includes model loading overhead (mitigated by model warming)

**Custom Batch Size** (advanced):
```python
reranker = NeuralReranker(model_name="unicamp-dl", batch_size=16)  # Manual override
```

## Performance

### Startup Time

**Index Loading** (with lazy loading enabled):
- **First run**: 30-60s (builds term offset cache)
- **Subsequent runs**: 1-2s (loads from `final/index_offsets.json` cache)
- **Speedup**: 30-60x faster after initial startup

The persistent offset cache is automatically created and reused, requiring no manual intervention.

### Query Time

**Without Reranking (BM25 only)**:
- Total: 50-200ms

**With Reranking (BM25 + Neural)**:
- BM25 retrieval: 50-200ms
- Neural reranking: 500-700ms (GPU) or 1000-1200ms (CPU)
- **Total: 700-900ms (GPU) or 1200-1400ms (CPU)**

**Optimizations Applied**:
- Model warming eliminates 500-1000ms model loading overhead
- Optimized content truncation (200 words) reduces inference time by ~20%
- Smart candidate selection skips reranking for small result sets (<50ms for ≤10 results)
- Automatic batch size optimization maximizes hardware utilization

### Quality Improvement Example

**Query**: "capital de portugal"

**BM25 Only** (lexical):
1. "Portugal" (contains both words)
2. "Capital" (contains keyword)
3. "Lisboa" ✅ (correct, but ranked lower)

**With Neural Reranking** (semantic):
1. "Lisboa" ✅ (understands intent)
2. "Portugal"
3. "Capitais da Europa"

## Troubleshooting

### Model Loading Failed

**Cause**: Missing dependencies or no internet connection (models download on first use)

**Solution**:
```bash
uv sync  # Install dependencies
```

### Slow Startup (First Run)

**Cause**: Building term offset index cache for the first time

**Solution**: This is expected behavior. The cache is saved to `wiki_index_full/final/index_offsets.json` and subsequent startups will be 30-60x faster (1-2s instead of 30-60s).

If the cache file is corrupted:
```bash
rm wiki_index_full/final/index_offsets.json  # Rebuild on next startup
```

### Reranking is Slow

**Solutions**:
- Use GPU if available (3-5x speedup)
- Reduce `num_candidates` (100 → 50)
- Use `unicamp-dl` instead of `mmarco` (faster model)
- Verify model warming is working (check logs for "Reranker loaded successfully")

### Out of Memory

**Solutions**:
- System automatically uses conservative batch sizes on CPU (8 vs 32)
- Manually reduce batch size: `NeuralReranker(batch_size=4)`
- Reduce `num_candidates`
- Use CPU instead of GPU (set `device="cpu"` explicitly)

### Cache Issues

**Problem**: Offset cache not loading or corrupted

**Solution**:
```bash
# Rebuild index cache
rm wiki_index_full/final/index_offsets.json

# Restart server - cache will rebuild automatically
```

## Advanced Features

### Model Warming Pattern

For production deployments, pre-load the reranker at application startup:

```python
from indexa.reranker import NeuralReranker

# At application startup
global_reranker = NeuralReranker(model_name="unicamp-dl")

# In request handler
def search_handler(query):
    results = engine.search_with_reranking(
        query=query,
        reranker=global_reranker  # Reuse pre-loaded model
    )
```

This pattern is automatically used by the API server for optimal performance.

### Performance Monitoring

Check logs for performance metrics:
```
INFO: Stage 1: BM25 retrieval for 'portugal' (top 100)
INFO: BM25 retrieved 100 candidates in 0.123s
INFO: Stage 2: Neural reranking with model 'unicamp-dl'
INFO: Auto-selected batch size: 32 (device: cuda)
INFO: Reranking complete in 0.687s
```

## References

- [MS MARCO Dataset](https://microsoft.github.io/msmarco/)
- [Cross-Encoders for Ranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [Unicamp-DL Portuguese Models](https://huggingface.co/unicamp-dl)
- [Sentence Transformers Documentation](https://www.sbert.net/)
