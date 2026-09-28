# Phase 1 Indexer Benchmark

This benchmark tests individual parameter impact on indexing performance.

## What it tests

Tests 4 parameters independently (23 total runs, ~8.5 hours):

1. **Indexer batch_size**: [250, 500, 1000, 1500, 2000, 3000] (6 runs)
2. **Block size (MB)**: [100, 150, 200, 250, 300, 400] (6 runs)
3. **Memory limit (MB)**: [1200, 1400, 1600, 1800, 2000] (5 runs)
4. **Tokenizer batch_size**: [1000, 2500, 5000, 7500, 10000, 15000] (6 runs)

Each parameter is varied while others stay at default values.

## Requirements

- Python environment with all indexer dependencies installed via `uv sync`
- Virtual environment at `.venv/` (created by uv)
- Sufficient disk space for temporary index directories
- Full dataset: `ptwiki-articles-with-redirects.arrow`

## Usage

```bash
# IMPORTANT: Run from the Assignment 1 base directory, not from benchmark/
cd "Assignment 1"

# Make sure dependencies are installed
uv sync

# Run the full benchmark (recommended to run overnight)
# The script automatically uses .venv/bin/python
python3 benchmark/benchmark_phase1.py

# The script will:
# - Run 23 configurations on the full dataset (1,154,228 docs)
# - Save results after each run to benchmark/benchmark_phase1_results.json
# - Clean up index directories between runs
# - Show progress with ETA
# - Can be interrupted (Ctrl+C) and resumed later
```

## Output

**File**: `benchmark/benchmark_phase1_results.json`

Each result includes:
- Configuration (all 4 parameter values)
- Which parameter was varied
- **Status**: `success`, `memory_guard_triggered`, `crashed`, `timeout`, or `error`
- **Error message** (if failed)
- Timing metrics (if successful):
  - tokenization_seconds
  - block_writing_seconds
  - merge_seconds
  - metadata_seconds
  - total_indexing_seconds
- Throughput (if successful):
  - documents_per_second
  - tokens_per_second
- Statistics (if successful):
  - documents_indexed
  - unique_terms
  - total_tokens
  - blocks_created
  - avg_document_length

## Features

- ✅ **Resume capability**: Can stop and resume - progress is saved after each run
- ✅ **Error detection**: Catches memory guard OOM kills, crashes, timeouts
- ✅ **Automatic cleanup**: Removes index directories after each run to save disk space
- ✅ **Progress tracking**: Shows ETA and completion time estimate
- ✅ **Detailed logging**: Records all configurations and results

## Example Results

```json
{
  "run_number": 1,
  "timestamp": "20251107_221545",
  "varied_parameter": "indexer_batch_size",
  "varied_value": 250,
  "configuration": {
    "indexer_batch_size": 250,
    "block_size_mb": 200,
    "memory_limit_mb": 1800,
    "tokenizer_batch_size": 5000,
    "min_term_frequency": 1
  },
  "status": "success",
  "timings": {
    "tokenization_seconds": 329.98,
    "block_writing_seconds": 97.02,
    "merge_seconds": 278.21,
    "metadata_seconds": 5.09,
    "total_indexing_seconds": 1325.21
  },
  "throughput": {
    "documents_per_second": 870.97,
    "tokens_per_second": 233968.83
  },
  "statistics": {
    "documents_indexed": 1154228,
    "unique_terms": 3842071,
    "total_tokens": 310059273,
    "blocks_created": 83,
    "avg_document_length": 268.62
  }
}
```

## Analyzing Results

After the benchmark completes, you can analyze the results:

```bash
# Pretty print results
python3 -m json.tool benchmark/benchmark_phase1_results.json | less

# Find fastest configuration
python3 -c "
import json
with open('benchmark/benchmark_phase1_results.json') as f:
    results = json.load(f)
successful = [r for r in results if r['status'] == 'success']
fastest = min(successful, key=lambda r: r['timings']['total_indexing_seconds'])
print(f\"Fastest: {fastest['varied_parameter']}={fastest['varied_value']}\")
print(f\"Time: {fastest['timings']['total_indexing_seconds']:.1f}s\")
"

# Check for failures
python3 -c "
import json
with open('benchmark/benchmark_phase1_results.json') as f:
    results = json.load(f)
failed = [r for r in results if r['status'] != 'success']
for r in failed:
    print(f\"{r['varied_parameter']}={r['varied_value']}: {r['status']} - {r.get('error_message', 'N/A')}\")
"
```

## Notes

- The benchmark uses the full dataset: `ptwiki-articles-with-redirects.arrow`
- Each run has a 2-hour timeout
- Memory guard kills (OOM) are detected and recorded
- Index directories are automatically cleaned up to save disk space
- Tokenizer config files are created temporarily for tokenizer_batch_size tests

## Next Steps

After Phase 1 completes:
1. Analyze which parameters have the biggest impact on speed
2. Identify best values for each parameter
3. Run Phase 2: test combinations of the best values
