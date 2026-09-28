# Forward Index Implementation

This document describes the SQLite-based forward index used for efficient document retrieval.

## Overview

The system uses SQLite as a forward index to enable O(1) document content lookups by doc_id during search operations. This replaced the previous JSON-based approach which required O(n) sequential scanning.

## Motivation

**Problem:** Original implementation used `documents.json` for storing document metadata, requiring full file scan for each document lookup during search.

**Solution:** SQLite database with indexed primary key for instant lookups.

**Performance gain:** 100-1000x faster document retrieval (100µs vs 10-100ms).

## Schema Design

**Database:** `documents.db` (SQLite 3)

```sql
CREATE TABLE IF NOT EXISTS documents (
    doc_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL
);
```

**Fields:**
- `doc_id`: Primary key matching inverted index document IDs
- `title`: Document title (used for display and URL generation)
- `content`: Full document text (for previews and similarity search)

**Index:** Automatic B-tree index on `doc_id` (PRIMARY KEY) enables O(1) lookups.

## Implementation

### Indexing Phase (spimi.py)

Documents are written to SQLite incrementally during indexing:

**Location:** `src/indexa/indexer/spimi.py`

**Process:**
1. Document buffer accumulates documents (1000 per batch)
2. Batch insert using `executemany()` for performance
3. Transaction committed every batch
4. Memory cleared immediately after write

**Code:**
```python
# Batch insert (1000 documents per transaction)
self.db_cursor.executemany(
    "INSERT OR REPLACE INTO documents (doc_id, title, content) VALUES (?, ?, ?)",
    self.doc_buffer,
)
self.db_connection.commit()
```

**Memory efficiency:** Documents streamed to disk, not accumulated in memory.

### Search Phase (document_store.py)

Fast document retrieval with caching layer.

**Location:** `src/indexa/searcher/document_store.py`

**Features:**
- O(1) SQLite lookups using PRIMARY KEY index
- In-memory LRU cache (max 1000 documents)
- Automatic Wikipedia URL generation from titles
- Document preview support (truncated content)

**Code:**
```python
# Primary key lookup (O(1) with B-tree index)
self.cursor.execute(
    "SELECT title, content FROM documents WHERE doc_id = ?",
    (doc_id,),
)
```

**Caching:** Frequently accessed documents cached in memory to avoid repeated database queries.

## Performance

**Lookup times:**
- Cache hit: ~1µs (in-memory dictionary)
- Cache miss: ~100µs (SQLite B-tree lookup)
- vs JSON scan: ~50-200ms (500-2000x slower)
- vs Arrow scan: ~10-100ms (100-1000x slower)

**Indexing overhead:**
- Batch inserts: ~44 seconds for 1.15M documents
- ~2.7% of total indexing time
- Minimal memory overhead (batched writes)

## Usage Example

```python
from pathlib import Path
from indexa.searcher.document_store import DocumentStore

# Initialize store
store = DocumentStore(db_path=Path("index/documents.db"))

# Retrieve document
doc = store.get_document(doc_id=42)
# Returns: {
#   "title": "Inteligência Artificial",
#   "content": "...",
#   "url": "https://pt.wikipedia.org/wiki/Inteligência_Artificial"
# }

# Get preview (truncated content)
preview = store.get_document_preview(doc_id=42, max_length=200)

# Close connection
store.close()
```

## Design Trade-offs

**Advantages:**
- Constant-time lookups regardless of collection size
- Standard format with robust tooling and support
- Built incrementally during indexing (memory efficient)
- Atomic transactions ensure consistency
- In-memory caching for hot documents

**Disadvantages:**
- Additional dependency (mitigated: sqlite3 is Python standard library)
- Slight indexing overhead (~2.7% of total time)
- Database file size (~same as JSON, smaller than Arrow)

**Alternative approaches considered:**
- JSON file: Simple but O(n) lookups unacceptable for search
- Arrow file: Requires full scan or complex indexing
- In-memory dict: Violates memory constraints for large collections
- Document store services: Over-engineered for this use case

## Integration Points

**Created by:** `SPIMIIndexer._flush_documents_to_sqlite()` in `src/indexa/indexer/spimi.py`

**Used by:**
- `DocumentStore.get_document()` in `src/indexa/searcher/document_store.py`
- `SearchEngine` for result enrichment
- API routes for document preview endpoints

**File location:** `<index_dir>/documents.db`

## Consistency

**Inverted index → Forward index mapping:**
- Document IDs in inverted index (`final/index.jsonl`) correspond exactly to `doc_id` in SQLite
- Both created during same indexing pass
- No synchronization issues or ID mismatches

**Transaction safety:**
- Batch commits ensure atomic writes
- Database recovered automatically on index rebuild
- No partial state issues

## Future Optimizations

Potential improvements (not currently implemented):

1. **Full-text search:** SQLite FTS5 extension for content search
2. **Compression:** BLOB compression for large documents
3. **Prepared statements:** Reuse compiled queries (minimal gain with caching)
4. **Write-ahead logging:** Enable WAL mode for concurrent reads during indexing
5. **Larger cache:** Increase cache size for large RAM systems

Current implementation prioritizes simplicity and proven reliability over marginal optimizations.
