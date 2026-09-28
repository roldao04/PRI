# Indexa - Setup and Usage Guide

This guide provides step-by-step instructions for setting up, running, and using the Indexa search engine.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Indexing Documents](#indexing-documents)
- [Running the Search API](#running-the-search-api)

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10 or higher**
- **uv** - Fast Python package installer and resolver
  - Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Or visit: <https://docs.astral.sh/uv/getting-started/installation/>

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd practical-assignments-pri_g01/Assignment\ 1/
```

### 2. Install Dependencies

```bash
uv sync
```

This will:

- Create a virtual environment automatically
- Install all required Python packages
- Download NLTK data (stopwords, stemmers)

### 3. Install Pre-commit Hooks (Optional)

If you plan to contribute to the project:

```bash
uv run pre-commit install
```

This ensures code quality checks run automatically before each commit.

## Quick Start

### Index a Sample Collection

```bash
# Index documents from a directory
uv run cli documents/ -o wiki_index_full/

# Index a specific Arrow file (Wikipedia dataset)
uv run cli ptwiki-articles-with-redirects.arrow -o index/
```

### Start the Search API
Search api will look for the index in the specified output directory "wiki_index_full/":

```bash
uv run uvicorn indexa.entrypoints.asgi:app --reload
```

Then open your browser to:

- Web Interface: <http://localhost:8000>
- API Documentation: <http://localhost:8000/docs>

## Indexing Documents

### Basic Usage

The CLI indexer is the main tool for building search indices:

```bash
uv run cli <input_path> -o <output_directory>
```

**Parameters:**

- `input_path`: Path to documents directory or Arrow file
- `-o, --output`: Directory where index will be saved

### Examples

**Index full Portuguese Wikipedia:**
```bash
uv run cli ptwiki-articles-with-redirects.arrow -o wiki_index_full/
```

**Index subset for testing:**
```bash
uv run cli ptwiki-articles-with-redirects.arrow -o test_index/ --max-docs 10000
```

**Index custom document directory:**
```bash
uv run cli documents/ -o wiki_index_full/ --language pt
```

### Output Structure

After indexing completes, the output directory contains:

```bash
index_output/
├── metadata.json       # Index statistics and configuration
├── documents.db        # SQLite forward index (doc_id -> content)
├── final/
│   └── index.jsonl     # Inverted index (term -> postings)
└── timing_stats.json   # Performance metrics (if instrumented)
```

**Key files:**
- `metadata.json`: Number of documents, terms, tokenizer configuration
- `documents.db`: SQLite database for fast document retrieval during search
- `final/index.jsonl`: Inverted index mapping terms to document postings

## Running the Search API

### Start the Server

```bash
uv run uvicorn indexa.entrypoints.asgi:app --reload
```

Options:

- `--reload`: Auto-reload on code changes (development)
- `--host 0.0.0.0`: Listen on all network interfaces
- `--port 8080`: Use custom port

### Production Server

For production deployment:

```bash
uv run uvicorn indexa.entrypoints.asgi:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

Once the server is running:

#### Search Endpoint

```bash
POST /api/v1/search/
Content-Type: application/json

{
  "query": "search terms",
  "top_k": 10
}
```

#### Health Check

```bash
GET /api/v1/healthcheck/
```

#### Interactive Documentation

Visit <http://localhost:8000/docs> for Swagger UI with interactive API testing.
