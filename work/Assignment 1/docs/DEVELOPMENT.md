# Indexa - Development Guide

This document contains development information and best practices for contributors to the Indexa search engine project.

## Prerequisites

- Python 3.10+
- uv (install from <https://docs.astral.sh/uv/getting-started/installation/>)

## Quick Setup

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Install pre-commit hooks (runs linting on every commit):

   ```bash
   uv run pre-commit install
   ```

## Project Structure

```bash
Assignment 1/
├── src/indexa/
│   ├── core/                   # Core utilities
│   │   ├── model.py            # Document data models (Pydantic)
│   │   ├── limit_memory.py     # Memory monitoring (2GB constraint)
│   │   └── logging.py          # Logging configuration
│   ├── indexer/                # Indexing pipeline
│   │   ├── spimi.py            # SPIMI indexer implementation
│   │   ├── tokenizer.py        # Text tokenization and normalization
│   │   ├── document_reader.py  # Document format readers
│   │   └── models.py           # Index data structures
│   └── entrypoints/            # Application entry points
│       ├── cli.py              # CLI for indexing
│       ├── api/                # REST API for search
│       └── asgi.py             # ASGI server
├── static_pages/               # Web interface
└── tests/                      # Unit and integration tests
```

## Running the Project

### Indexing Documents

Index a document collection with the CLI:

```bash
# Basic usage
uv run cli input_documents/ -o index_output/

# With custom settings
uv run cli input_documents/ -o index_output/ \
  --language pt \
  --min-token 3 \
  --min-term-freq 2 \
  --memory-limit 1800
```

The indexer will respect the 2GB memory constraint through the SPIMI algorithm.

### Running the Search API

Start the FastAPI server:

```bash
uv run uvicorn indexa.entrypoints.asgi:app --reload
```

Access:

- **Web Interface**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs`

## Development Best Practices

### Code Quality

This project enforces strict code quality standards:

- **Type Checking**: Full type annotation coverage with Pyright
- **Linting**: Ruff with comprehensive rule set (100-char line limit)
- **Formatting**: Automatic code formatting with Ruff
- **Import Sorting**: Organized imports with isort integration

### Git Workflow

1. **Make frequent commits**: Pre-commit hooks run type checking and linting automatically
2. **Follow the enforced standards**: The project uses strict linting rules for consistency
3. **Test before committing**: All code is validated before entering the repository

### Skipping Pre-commit Hooks

⚠️ **NOT RECOMMENDED!**

This repository includes pre-commit hooks that verify and standardize code before committing. However, if you're close to a deadline and need to bypass them temporarily:

```bash
git commit --no-verify -m "Your commit message"
```
