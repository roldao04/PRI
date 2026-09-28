# Indexa - Architecture Documentation

High-level architectural decisions and design patterns. For implementation details and usage, see [README.md](../README.md).

## System Architecture

```text
┌─────────────────────────────────────────────────┐
│              Entry Points                       │
│  CLI Indexer (cli.py)  │  FastAPI Server       │
└────────────┬────────────┴───────────┬───────────┘
             │                        │
             ▼                        ▼
┌────────────────────────┐  ┌────────────────────┐
│  Indexing Pipeline     │  │   Search API       │
│  - Document Reader     │  │   - Query Parser   │
│  - Tokenizer (PT)      │  │   - BM25 Ranker    │
│  - SPIMI Indexer       │  │   - Result Builder │
└────────────┬───────────┘  └─────────┬──────────┘
             │                        │
             ▼                        ▼
┌─────────────────────────────────────────────────┐
│            Inverted Index                       │
│   (JSONL final, msgpack intermediate)          │
└─────────────────────────────────────────────────┘
```

## Core Design Decisions

### 1. SPIMI Algorithm

**Decision:** Use Single-Pass In-Memory Indexing with batch processing

**Why:**

- Memory-constrained (2GB limit) - builds partial blocks, not full index
- Single pass - no expensive global sorting phase like BSBI
- Batch optimized - 1000 docs, 50000 terms per batch

**Trade-off:** More complex than naive approaches, but necessary for memory constraint

---

### 2. Positional Indexing

**Decision:** Store term positions in postings list

**Why:**

- Enables phrase queries ("search engine")
- Supports proximity search
- Required for BM25 term frequency calculation

**Trade-off:** 2-3x larger index, but essential for quality search

---

### 3. Portuguese-Only Tokenization

**Decision:** Optimize for Portuguese Wikipedia corpus only

**Why:**

- pystemmer C-based Snowball: 10-50x faster than NLTK Python
- Simpler, more focused implementation
- Better quality through language-specific optimization

**Trade-off:** Not multilingual, but better performance for target use case

---

### 4. Hybrid Serialization Strategy

**Decision:** msgpack for intermediate blocks, JSONL for final index

**Why:**

- **Intermediate (msgpack):** Binary format is 2-3x faster for frequent I/O during indexing
- **Final (JSONL):** Human-readable for debugging, line-by-line loading for search

**Trade-off:** Complexity of two formats, but optimizes both indexing speed and debuggability

---

### 5. Multi-Level Batch Processing

**Decision:** Batch processing at document, token, and term levels

**Implementation:**

- Documents: 1000 per batch (I/O optimization)
- Tokens: 5000 per pystemmer call (amortize overhead)
- Terms: 50000 per index insertion (cache locality)

**Why:** Reduces function call overhead, improves CPU cache utilization

**Result:** 3-5x overall speedup

---

### 6. Larger Block Threshold

**Decision:** 200MB blocks (vs typical 100MB)

**Why:**

- Fewer blocks to merge (10 vs 20 blocks)
- Still safe under 2GB with monitoring
- Faster final merge phase

**Trade-off:** Larger memory peaks, but monitoring ensures safety

---

### 7. Sampling-Based Memory Estimation

**Decision:** Sample 10% of terms for memory size estimation

**Why:**

- 5-10x faster than full iteration
- 95%+ accuracy sufficient for block threshold decisions
- Reduces overhead during indexing

**Trade-off:** Slight inaccuracy, but performance gain is worth it

---

### 8. SQLite Forward Index

**Decision:** Use SQLite database for forward index (doc_id to document content mapping)

**Why:**

- O(1) lookups via B-tree index vs O(n) sequential JSON scan
- Built incrementally during indexing (batch inserts, 1000 docs per transaction)
- Standard library (no external dependencies)
- Enables in-memory caching for frequently accessed documents

**Trade-off:** Slight indexing overhead (2.7% of total time), but 100-1000x faster document retrieval during search

## Design Patterns

### Factory Pattern

**Where:** `DocumentReaderFactory`, `TokenizerFactory`

**Why:** Encapsulates format-specific logic, easy to extend with new formats

---

### Strategy Pattern

**Where:** `TokenizerConfig`

**Why:** Configure behavior (stemming, stop words) without subclassing

---

### Iterator Pattern

**Where:** `ArrowDocumentReader.read_documents()`, `tokenize_with_positions()`

**Why:** Memory-efficient streaming, lazy evaluation

---

### Template Method Pattern

**Where:** `DocumentReader` abstract base class

**Why:** Define `read()` contract, subclasses implement format-specific logic

## Performance Optimizations

### Evolution from Initial to Optimized Implementation

| Component | Initial | Optimized | Speedup |
|-----------|---------|-----------|---------|
| **Tokenizer** | NLTK Python stemmer | pystemmer + batch (5000) | 10-50x |
| **Index Building** | Individual insertions | Sorted batch (50000) | 2-3x |
| **I/O** | JSON for all | msgpack for blocks | 2-3x |
| **Memory Check** | Every document | Every 10k docs + sampling | 5-10x |
| **Block Size** | 100MB (20 blocks) | 200MB (10 blocks) | 2x merge speed |
| **Doc Retrieval** | JSON scan | SQLite B-tree | 100-1000x |

**Overall Result:** 3-5x faster indexing while maintaining 2GB memory constraint, 100-1000x faster document retrieval during search

## Key Trade-offs

1. **Positional index**: 2-3x larger size for phrase queries and better ranking
2. **Portuguese-only**: Simplicity and performance over multilingual support
3. **Batch processing**: Complexity for 3-5x speed improvement
4. **Hybrid formats**: Two serialization strategies for optimal performance in each phase
5. **Larger blocks**: Higher memory peaks but faster merging (safe with monitoring)
6. **Sampling estimation**: ~5% accuracy loss for 5-10x speed gain
7. **SQLite forward index**: Slight indexing overhead for 100-1000x faster document retrieval

---

**For implementation details, usage instructions, and optimization benchmarks, see:**

- [README.md](../README.md) - Complete technical report
- [SETUP.md](SETUP.md) - Installation and usage
- [FORWARD_INDEX.md](FORWARD_INDEX.md) - SQLite forward index implementation
- [SEARCH_ENGINE_GUIDE.md](SEARCH_ENGINE_GUIDE.md) - Search and ranking implementation
