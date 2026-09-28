# Migration from Assignment 1 to Assignment 2

This document describes what was migrated from Assignment 1 and what changes were made for Assignment 2.

## What Was Copied from Assignment 1

Assignment 2 is built on top of Assignment 1's codebase. The following components were copied:

### Core Functionality
- **Complete source code** (`src/indexa/`):
  - SPIMI indexer with positional indexing
  - Portuguese tokenizer (pystemmer-based with batch processing)
  - BM25F search engine with field-weighted scoring
  - SQLite forward index for fast document retrieval
  - Multiple similarity methods (cosine, Jaccard, hybrid, field-weighted)
  - REST API with FastAPI
  - CLI for indexing

### Supporting Files
- **Project configuration** (`pyproject.toml`): Dependencies and build configuration
- **Web interface** (`static_pages/`): HTML/CSS/JS search interface
- **Documentation** (`docs/`): Architecture, optimization, and setup guides
- **Pre-built index** (`wiki_index_full/`): Portuguese Wikipedia index (1.15M documents)

### What Was **NOT** Copied
- Benchmark code (`benchmark/`)
- Other index variants (`wiki_index_100000/`, etc.)
- Python cache files (`__pycache__/`)
- Lock files (`uv.lock`)
- Source data files (`.arrow` files)

## Key Changes for Assignment 2

### 1. Removed 2GB Memory Constraint

**Rationale**: Assignment 1 had a strict 2GB memory limit as a core requirement. Assignment 2 focuses on adding AI capabilities (neural reranking, RAG) rather than memory optimization, so this constraint was removed to simplify development.

**Changes Made**:

#### `src/indexa/core/limit_memory.py`
- **Gutted the memory monitor**: All monitoring functions are now no-ops
- `start_memory_monitor()`: Logs a message but doesn't start monitoring thread
- `get_current_memory_usage_in_mb()`: Returns 0
- Existing code still works (function signatures preserved), but no enforcement

#### `src/indexa/entrypoints/cli.py`
- **Memory monitor disabled**: `start_memory_monitor()` called with `show_memory_updates=False`
- **Validation removed**: No longer exits if `--memory-limit > 2000MB`
- **Increased defaults**:
  - `--memory-limit`: 1800MB → **8000MB** (default, no hard limit)
  - `--block-size`: 120MB → **500MB** (fewer blocks, faster merging)
- **Updated help text**: Removed mentions of 2GB constraint

#### `src/indexa/indexer/spimi.py`
- **No changes needed**: Already doesn't directly call memory monitoring functions

### 2. Performance Implications

With memory constraints removed:

**Benefits**:
- Larger block sizes → Fewer merge operations → Faster indexing
- Can use more aggressive in-memory caching
- Simpler code without monitoring overhead

**Trade-offs**:
- May use more memory (but that's acceptable for Assignment 2 focus)
- No automatic crash protection if memory usage spirals

### 3. All Assignment 1 Features Preserved

Everything else from Assignment 1 remains intact:
- SPIMI algorithm (still used for disk-based indexing)
- Batch tokenization (5000 tokens per pystemmer call)
- BM25F ranking with field weights
- SQLite forward index
- Multiple similarity methods
- FastAPI REST endpoints
- Web search interface

## What's Next for Assignment 2

This migration provides the foundation for adding AI capabilities:

1. **Neural Reranking** (10 points)
   - Add cross-encoder model for semantic reranking
   - Two-stage pipeline: BM25 → Neural rerank

2. **Answer Generation / RAG** (7 points)
   - Integrate LLM (Google Gemini API)
   - Generate natural language answers from retrieved documents

3. **Optional Enhancements** (3 points)
   - Semantic snippet extraction
   - Embedding-based similarity
   - Query expansion with LLM

## Compatibility Notes

- **Index reuse**: The existing `wiki_index_full/` from Assignment 1 is fully compatible
- **API compatibility**: All Assignment 1 endpoints work unchanged
- **Dependencies**: Same as Assignment 1 (new dependencies will be added for AI features)

## Summary

Assignment 2 preserves all the sophisticated engineering from Assignment 1 (SPIMI, batch processing, BM25F, SQLite indexing) while removing the artificial 2GB memory constraint to focus on AI enhancements. This provides a solid foundation for building a modern semantic search system.
