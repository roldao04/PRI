# Indexa - Search Engine Technical Report

A memory-constrained search engine using SPIMI (Single-Pass In-Memory Indexing) for Portuguese Wikipedia with 3-5x performance optimizations.

## Quick Links

- **[Setup Guide](docs/SETUP.md)** - Installation and usage
- **[Architecture](docs/ARCHITECTURE.md)** - Design decisions and patterns
- **[Forward Index](docs/FORWARD_INDEX.md)** - SQLite forward index implementation
- **[Search Engine](docs/SEARCH_ENGINE_GUIDE.md)** - Search and ranking implementation
- **[Optimization Journey](docs/OPTIMIZATION_JOURNEY.md)** - Performance optimization chronicle
- **[Development](docs/DEVELOPMENT.md)** - Development workflow

---

## 1. Document Ingestion

### How Documents Are Ingested

Documents are processed using a **streaming architecture** to minimize memory:

- **Sequential Processing**: Documents read one at a time, never loading entire collection
- **Batch Processing**: Apache Arrow files processed in batches of 1,000 documents (optimized)
- **Incremental Indexing**: Terms extracted and buffered (50,000 terms) before adding to index block

### 2GB Memory Constraint Compliance

The system respects the 2GB constraint through:

1. **Memory Monitoring** ([limit_memory.py](src/indexa/core/limit_memory.py)):
   - Active monitoring thread tracking process memory
   - Terminates if exceeds 2GB limit
   - Real-time usage feedback

2. **SPIMI Algorithm** ([spimi.py](src/indexa/indexer/spimi.py)):
   - Builds partial index blocks in memory (200MB per block - optimized)
   - Writes blocks to disk when threshold reached
   - Memory limit: 1800MB (safe buffer below 2GB)
   - Blocks cleared immediately after writing

3. **Streaming Readers** ([document_reader.py](src/indexa/indexer/document_reader.py)):
   - Iterator-based reading (one document at a time)
   - No full dataset in memory
   - Automatic redirect filtering for Wikipedia

### Supported Formats

- **Apache Arrow** (`.arrow`, `.feather`) - Primary format for Portuguese Wikipedia
- **Plain Text** (`.txt`, `.md`) - With automatic encoding detection

---

## 2. Tokenizer Characterization

### Design

The tokenizer ([tokenizer.py](src/indexa/indexer/tokenizer.py)) implements a **batch-optimized** linguistic pipeline:

### Processing Pipeline

1. **Token Extraction**:
   - Regex: `[a-zA-ZÀ-ÿ]+` (supports Portuguese accents)
   - Pre-compiled at module level for performance

2. **Token Filtering**:
   - Min length: 3 characters (configurable)
   - Max length: 50 characters (configurable)
   - Stop word removal (NLTK Portuguese corpus)

3. **Stemming** (Performance Optimized):
   - Uses `pystemmer` (C-based Snowball stemmer)
   - **Batch stemming**: Processes 5,000 tokens at once
   - **10-50x faster** than NLTK Python-based stemmer
   - Portuguese Snowball algorithm

#### Comprehensive Benchmark Results pystemmer vs NLTK

| Words Tested | pystemmer (s) | NLTK (s) | Speedup |
|-------------|---------------|-----------|---------|
| 1,000       | 0.0002       | 0.0032   | **15.2x** |
| 5,000       | 0.0008       | 0.0164   | **20.1x** |
| 10,000      | 0.0017       | 0.0319   | **19.3x** |
| 25,000      | 0.0037       | 0.0898   | **24.1x** |

### Configuration
We save in tokenizer_config.json the following configuration to later use in searching/ranking:

```python
TokenizerConfig(
    use_stemming=True,
    use_stopwords=True,
    custom_stopwords: list[str] | None = None,
    min_token_length=3,
    max_token_length=50,
    batch_size=5000,  # Batch stemming size, deprecated
)
```


### Positional Indexing

- `tokenize_with_positions()` returns (term, position) tuples
- Enables phrase queries and proximity search
- Optimized with batch processing

---

## 3. Index Format

### Storage Structure

```bash
index_output/
├── metadata.json       # Index statistics and configuration
├── documents.db        # SQLite forward index (doc_id -> content)
├── blocks/             # Temporary (msgpack binary - deleted after merge)
│   └── block_*.msgpack
└── final/
    └── index.jsonl     # Final inverted index (JSONL for compatibility)
```

### Index Format (JSONL)

Each line in `final/index.jsonl`:

```json
{"term": "portugal", "postings": [{"doc_id": 0, "positions": [5, 12, 45]}, {"doc_id": 5, "positions": [2, 89]}]}
```

### Metadata (`metadata.json`)
For documentation and debugging:

```json
{
  "num_documents": 1154228,
  "num_terms": 3842071,
  "num_blocks": 109,
  "avg_doc_length": 268.6,
  "tokenizer_config": {...}
}
```

### Forward Index (`documents.db`)

SQLite database for O(1) document retrieval:

```sql
CREATE TABLE documents (
    doc_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL
);
```

**Purpose:**
- Fast document lookups during search (100-1000x faster than JSON)
- Built incrementally during indexing (batch inserts, 1000 docs per transaction)
- In-memory caching layer for frequently accessed documents

**Performance:**
- Lookup time: ~100µs (with caching: ~1µs)
- vs JSON scan: ~50-200ms per lookup

See [docs/FORWARD_INDEX.md](docs/FORWARD_INDEX.md) for implementation details.

### Design Decisions

We use a hybrid approach optimized for different use cases:

**Why SQLite for forward index?**

- O(1) indexed lookups (B-tree) vs O(n) sequential scan
- Incremental building during indexing (memory efficient)
- Standard library (no external dependencies)
- In-memory caching for hot documents

**Why JSONL for inverted index?**

- Human readable (easy to debug)
- Language agnostic (any language can read)
- Streaming-friendly (line-by-line reading)
- Backward compatible with existing search code

**Why msgpack for intermediate blocks?**

- 2-3x faster I/O than JSON
- Smaller file size (30-50% reduction)
- Temporary files (deleted after merge)
- Performance benefit without breaking compatibility


---

## 4. Performance Optimizations

Applied optimizations achieve **3-5x speedup** while maintaining full backward compatibility:

### Key Optimizations

1. **Binary Format for Blocks** (2-3x I/O speedup)
   - Temporary blocks use msgpack
   - Final index stays JSONL (compatible)

2. **Batch Processing** (2-3x speedup)
   - 50,000 term buffer before flushing
   - Terms sorted for better cache locality
   - Reduced function call overhead

3. **Optimized Tokenizer** (10-50x stemming speedup)
   - Batch `stemWords()` instead of individual calls
   - Pre-compiled regex at module level
   - 5,000 token batch size

4. **Smart Memory Management** (5-10x faster checks)
   - 10% sampling for memory estimation
   - Block size: 200MB (optimized for fewer merges)
   - Doc batch: 1,000 documents
   - Memory checks: every 10K docs (reduced overhead)

5. **SQLite Forward Index** (100-1000x faster document retrieval)
   - O(1) lookups vs O(n) JSON/Arrow scanning
   - Batch inserts (1000 docs per transaction)
   - In-memory caching for frequently accessed documents

### Performance Results

| Component | Old | New | Speedup |
|-----------|-----|-----|---------|
| Block I/O | JSON | msgpack | 2-3x |
| Stemming | Individual | Batch | 10-50x |
| Memory checks | Full scan | 10% sample | 5-10x |
| Doc retrieval | JSON scan | SQLite | 100-1000x |
| **Overall** | Baseline | **Optimized** | **3-5x** |

---

## 5. Dataset Statistics

**Portuguese Wikipedia**: `ptwiki-articles-with-redirects.arrow`

| Metric | Value |
|--------|-------|
| Total rows | 2,713,450 |
| Redirect pages | 1,559,196 (filtered) |
| **Indexed articles** | **1,154,228** |
| **Unique terms** | **3,842,071** |
| **Total tokens processed** | **310,059,273** |
| **Average document length** | **268.6 tokens** |
| **Blocks created** | **109** |

---

## 6. SPIMI Algorithm

### Phase 1: Block Creation

```bash
FOR each document batch (1000 docs):
  1. Tokenize with batch stemming (5000 tokens)
  2. Buffer terms (50,000 term buffer)
  3. Flush to index block when buffer full
  4. IF block >= 200MB:
       - Write block to disk (msgpack)
       - Clear from memory
       - Start new block
```

### Phase 2: Block Merging

```bash
1. Open all blocks (streaming)
2. K-way merge using min-heap
3. Merge postings for same terms
4. Write final index (JSONL format)
5. Delete temporary blocks
```

### Advantages

- **Memory Efficient**: Single block in memory
- **Scalable**: Handles collections larger than RAM
- **Fast**: Single-pass, no global sort
- **Optimized**: 3-5x faster than baseline

---

## 7. Usage

### Index Documents

```bash
# Full Portuguese Wikipedia
uv run cli ptwiki-articles-with-redirects.arrow -o wiki_index/

# Test with subset
uv run cli ptwiki-articles-with-redirects.arrow -o test_index/ --max-docs 10000

# Custom directory
uv run cli documents/ -o index/ --language pt
```

See [SETUP.md](docs/SETUP.md) and [USAGE.md](docs/USAGE.md) for complete usage guides.

---

## 8. Actual Performance (Final Test Results)

**Full Portuguese Wikipedia Indexing:**

| Metric | Value |
|--------|-------|
| **Documents indexed** | **1,154,228** |
| **Total time** | **1,925.45 seconds (32.1 minutes)** |
| **Throughput** | **599.5 documents/second** |
| **Peak memory usage** | **1,682.57 MB** |
| **Unique terms** | **3,842,071** |
| **Total tokens** | **310,059,273** |
| **Average doc length** | **268.6 tokens** |
| **Blocks created** | **109** |

### Performance by Scale

| Documents | Time | Memory | Throughput |
|-----------|------|--------|------------|
| 10,000 | ~17 seconds | < 500 MB | ~600 docs/sec |
| 100,000 | ~2.8 minutes | < 1 GB | ~600 docs/sec |
| 1,000,000 | ~27.8 minutes | < 1.7 GB | ~600 docs/sec |
| **1,154,228 (full)** | **~32.1 minutes** | **~1.68 GB** | **~600 docs/sec** |

---

## 9. Technical Details

### Dependencies

- **pyarrow**: Efficient Arrow file reading
- **pystemmer**: Fast C-based Portuguese stemming
- **msgpack**: Binary serialization for blocks
- **nltk**: Stop words and linguistic data
- **fastapi**: REST API framework
- **sqlite3**: Forward index for fast document retrieval (Python standard library)

### Memory Safety

- Process terminates if > 2GB
- Configurable limits (default 1800MB)
- Streaming architecture throughout
- No large in-memory structures

### Code Quality

- Full type hints (Pyright basic mode)
- Automated linting (Ruff)
- Modular, testable design
- Clear separation of concerns

---

## 10. Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| **SPIMI Algorithm** | Natural fit for 2GB constraint, efficient single-pass |
| **Positional Index** | Enables phrase queries, supports BM25 ranking |
| **SQLite Forward Index** | O(1) document retrieval, incremental building, standard format |
| **JSONL Final Format** | Human readable, compatible, language agnostic |
| **msgpack Blocks** | Fast I/O without breaking compatibility |
| **Batch Processing** | Reduces overhead, better cache utilization |
| **Streaming Architecture** | Respects memory limits, scales to large datasets |
| **Arrow Format** | Efficient columnar storage for Wikipedia |

---

## Conclusion

Indexa demonstrates a production-ready approach to memory-constrained search engine indexing. The optimized SPIMI implementation achieves **3-5x speedup** while maintaining full backward compatibility and respecting the 2GB memory constraint.

**Final Results:** The system successfully indexed **1,154,228 Portuguese Wikipedia articles** in **32.1 minutes** at a sustained throughput of **599.5 documents/second**, with peak memory usage of **1,682.57 MB** (well under the 2GB limit). The resulting index contains **3,842,071 unique terms** extracted from **310 million tokens**.

---

## Documentation

- **[docs/SETUP.md](docs/SETUP.md)** - Installation and setup instructions
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Design decisions and patterns
- **[docs/FORWARD_INDEX.md](docs/FORWARD_INDEX.md)** - SQLite forward index implementation
- **[docs/SEARCH_ENGINE_GUIDE.md](docs/SEARCH_ENGINE_GUIDE.md)** - Search and ranking implementation
- **[docs/OPTIMIZATION_JOURNEY.md](docs/OPTIMIZATION_JOURNEY.md)** - Performance optimization chronicle
- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** - Development workflow and guidelines
