# Finding Similar Documents

## Overview

The search engine implements relevance feedback through a "find similar documents" feature that returns documents similar to a given document. This is useful for "more like this" functionality.

**API Endpoint**: `GET /search_similar?doc_id={id}&similarity_method={method}`

## Similarity Methods

The system supports 5 similarity algorithms:

### 1. Hybrid (Default, Recommended)
- Combines TF-IDF cosine similarity (70%) and Jaccard similarity (30%)
- Best balance between semantic similarity and term overlap
- Jaccard scores are scaled to match TF-IDF range before combining

### 2. Cosine
- TF-IDF cosine similarity using unnormalized dot product
- Produces unbounded scores (typically 0-50+) matching BM25 range
- Fast and provides good quality results

### 3. Field-Weighted
- Computes separate similarities for each document field (title, heading, lead, body)
- Field weights: `title=3.0`, `heading=2.0`, `lead=2.0`, `body=1.0`
- Best for finding documents with similar structure
- May be slower on first use

### 4. Jaccard
- Set-based similarity: `|intersection| / |union|`
- Measures term overlap between documents
- Very fast, scores between 0 and 1
- Useful for finding documents with similar vocabulary

### 5. BM25 (Legacy)
- Extracts top 50 terms by TF-IDF and uses them as a pseudo-query
- Runs standard BM25 search with these terms
- Maintained for backward compatibility

## How It Works

### 1. Document Vector Extraction
```
source_doc → term frequencies {term1: freq1, term2: freq2, ...}
```

For lazy loading, uses approximate vectors (top 1000 terms) for speed.

### 2. Candidate Selection
Finds all documents sharing at least one term with the source document:
```
for each term in source_doc:
    get documents containing term
    build candidate vectors incrementally
```

This optimization avoids expensive full-vector construction for all documents.

### 3. Similarity Scoring

**TF-IDF Cosine**:
```
1. Compute TF-IDF vectors: tf-idf(term) = (1 + log(tf)) × log(N / df)
2. Calculate dot product: score = Σ(source_tfidf[term] × candidate_tfidf[term])
3. Scores are unbounded (similar to BM25 philosophy)
```

**Jaccard**:
```
similarity = |shared_terms| / |all_unique_terms|
```

**Hybrid**:
```
1. Compute both TF-IDF and Jaccard scores
2. Scale Jaccard [0,1] to TF-IDF range [0, max_tfidf]
3. Combine: score = (tfidf × 0.7) + (scaled_jaccard × 0.3)
```

### 4. Ranking and Results
- Sort by score (descending)
- Remove source document from results
- Return top N documents with metadata

## Key Optimizations

### Lazy Loading with Approximate Vectors
- Builds vectors from cached terms (no disk I/O)
- Samples vocabulary systematically if needed (~2000 terms)
- Ranks by TF-IDF and takes top 1000 terms
- Much faster than full vector construction

### Incremental Candidate Vectors
- Builds candidate vectors while loading postings
- Only includes terms seen during search (shared terms)
- Avoids N expensive `get_document_vector()` calls

### TF-IDF Weighting
- Uses logarithmic term frequency: `1 + log(tf)`
- IDF weighting: `log(N / df)`
- Emphasizes discriminative terms over common ones

## Implementation

**Main Files**:
- `src/indexa/searcher/search_engine.py` (lines 571-679): Core similarity search
- `src/indexa/searcher/search_engine.py` (lines 57-419): Similarity calculator with all methods
- `src/indexa/entrypoints/api/routes/search.py` (lines 125-208): API endpoint

**Entry Point**: `SearchEngine.search_similar(doc_id, num_results, min_score, similarity_method)`
