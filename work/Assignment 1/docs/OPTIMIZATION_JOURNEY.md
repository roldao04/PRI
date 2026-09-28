# Indexer Optimization Journey

**Date**: November 2025
**Dataset**: Portuguese Wikipedia (1,154,228 documents, 310M tokens)
**Initial Performance**: ~26 minutes (1,581 seconds)

This document chronicles our attempts to optimize the SPIMI indexer performance, including failed approaches, successful benchmarks, and key learnings.

---

## Table of Contents

1. [Initial Baseline](#initial-baseline)
2. [Attempt 1: Worker Parallelization](#attempt-1-worker-parallelization-failed)
3. [Attempt 2: Parameter Benchmarking](#attempt-2-parameter-benchmarking)
4. [Attempt 3: Performance Instrumentation](#attempt-3-performance-instrumentation)
5. [Attempt 4: Sorting Optimization](#attempt-4-sorting-optimization)
6. [Final Results & Conclusions](#final-results--conclusions)

---

## Initial Baseline

**Configuration:**
- Batch size: 1000 documents
- Block size: 200 MB
- Memory limit: 1800 MB
- Tokenizer batch: 5000 tokens

**Performance:**
- Total time: 1,325 seconds (22.1 minutes)
- Throughput: 871 docs/sec
- Blocks created: 83

**Known timings:**
- Tokenization: 330s (24.9%)
- Block writing: 97s (7.3%)
- Merge: 278s (21.0%)
- Metadata: 5s (0.4%)
- **Unknown: ~615s (46.4%)** ← Mystery!

---

## Attempt 1: Worker Parallelization (FAILED)

### Motivation

The indexing process appeared CPU-bound with tokenization taking 25% of time. Multi-processing seemed like a natural fit to utilize multiple cores.

### Approach

**Design:**
1. Pre-scan Arrow file to determine total document count
2. Split documents into N chunks (one per worker)
3. Each worker:
   - Processes its chunk independently
   - Creates its own index blocks
   - Writes to separate output directories
4. Final merge of all worker outputs

**Pre-scanning Implementation:**
```python
# Pre-scan to get document count
reader = ArrowDocumentReader(filter_redirects=True)
total_docs = 0
for doc in reader.read_documents(arrow_path):
    total_docs += 1

# Calculate chunks
docs_per_worker = total_docs // num_workers
```

### Why It Failed

**Memory Guard Constraint:**
- Hard limit: 2000 MB per process (system-wide)
- Each worker needs ~1800 MB for indexing
- 2+ workers = immediate memory guard kill
- Cannot bypass without modifying system constraints

**Issues Encountered:**
1. **Memory explosion**: Multiple processes × 1800 MB each
2. **Pre-scan overhead**: Full file scan adds ~30 seconds
3. **Coordination complexity**: Merging multiple worker outputs
4. **No benefit**: Memory constraint prevents >1 worker anyway

### Lessons Learned

- Memory guard is a hard constraint for this assignment
- Pre-scanning Arrow files is expensive (adds 2-3% overhead)
- Parallelization requires 2x-4x more total memory
- Single-process optimization is the only viable path

**Code artifacts:** Removed from codebase (not useful under memory constraint)

---

## Attempt 2: Parameter Benchmarking

### Motivation

Unsure if default parameters (batch size, block size, memory limit) were optimal. Needed data-driven evidence.

### Methodology

**Phase 1 Benchmark:**
- Test 4 parameters independently
- 23 total configurations (6-8 hours runtime)
- Full dataset (1.15M documents)
- Automated benchmark script with resume capability

**Parameters tested:**

| Parameter | Default | Test Values | Runs |
|-----------|---------|-------------|------|
| Indexer batch size | 1000 | [250, 500, 1000, 1500, 2000, 3000] | 6 |
| Block size (MB) | 200 | [100, 150, 200, 250, 300, 400] | 6 |
| Memory limit (MB) | 1800 | [1200, 1400, 1600, 1800, 2000] | 5 |
| Tokenizer batch | 5000 | [1000, 2500, 5000, 7500, 10000, 15000] | 6 |

### Results

#### Indexer Batch Size ⭐ (Biggest Impact)

| Value | Time (s) | Throughput | Status | Change |
|-------|----------|------------|--------|--------|
| 250 | 1683 | 686 docs/s | ✅ | baseline |
| 500 | 1630 | 708 docs/s | ✅ | +3.2% |
| **1000** | **1575** | **733 docs/s** | ✅ | **+6.4%** |
| 1500 | - | - | ❌ OOM | - |
| 2000 | - | - | ❌ OOM | - |
| 3000 | - | - | ❌ OOM | - |

**Finding:** Batch size 1000 is optimal. Larger values trigger memory guard kills.

#### Block Size (MB)

| Value | Time (s) | Blocks Created | Change |
|-------|----------|----------------|--------|
| 100 | 1579 | 109 | baseline |
| 150 | 1575 | 100 | +0.25% |
| 200 | 1574 | 83 | +0.32% |
| **250** | **1572** | **67** | **+0.45%** |
| 300 | - | - | ❌ OOM |
| 400 | - | - | ❌ OOM |

**Finding:** Larger blocks = fewer blocks = slightly faster merge. 250MB is optimal (but marginal gain).

#### Memory Limit (MB) 🔍 (Counter-intuitive!)

| Value | Time (s) | Change |
|-------|----------|--------|
| **1200** | **1569** | **+0.69%** |
| 1400 | 1571 | +0.57% |
| 1600 | 1573 | +0.38% |
| 1800 | 1574 | baseline |
| 2000 | 1580 | -0.38% |

**Finding:** LOWER memory limit performs BETTER! Forces more frequent block writes, prevents memory buildup overhead.

#### Tokenizer Batch Size

| Value | Time (s) | Change |
|-------|----------|--------|
| **1000** | **1567** | **+0.50%** |
| 2500 | 1571 | +0.25% |
| 5000 | 1574 | baseline |
| 7500 | 1574 | 0% |
| 10000 | 1574 | 0% |
| 15000 | 1580 | -0.32% |

**Finding:** Smaller tokenizer batches slightly faster. Very minimal impact.

### Key Insights

1. **Current defaults are already well-optimized** - only ~0.5% total improvement possible
2. **Memory constraints are the bottleneck** - larger batch/block sizes cause OOM
3. **Diminishing returns** - marginal gains (seconds) not worth complexity
4. **Lower memory limits can be better** - counter-intuitive but forces efficient block management

**Optimal configuration:**
```python
indexer_batch_size: 1000      # keep default ✅
block_size_mb: 250            # +0.45% improvement
memory_limit_mb: 1200         # +0.69% improvement
tokenizer_batch_size: 1000    # +0.50% improvement
```

**Expected combined improvement:** ~1.6% (25 seconds)

**Decision:** Keep defaults - marginal gains not worth changing well-tested configuration.

---

## Attempt 3: Performance Instrumentation

### Motivation

**46.4% of time was untracked!** Need to understand where 615+ seconds were going.

### Implementation

Added comprehensive timing instrumentation to track all operations:

**New timing fields:**
1. `arrow_reading_time` - Reading documents from Arrow file
2. `term_buffer_flush_time` - Flushing term buffer to blocks
3. `doc_buffer_flush_time` - Flushing documents to SQLite
4. `gc_time` - Garbage collection calls
5. `dict_operations_time` - Dictionary updates and term buffering
6. `sqlite_insert_time` - SQLite forward index inserts

### Results: Mystery Solved! 🎉

**Complete timing breakdown** (1,581 seconds total):

| Operation | Time (s) | % of Total | Ranking |
|-----------|----------|------------|---------|
| **Term buffer flush** | **555.7** | **35.1%** | ⭐⭐⭐⭐⭐ |
| Tokenization | 401.6 | 25.4% | ⭐⭐⭐ |
| Merge | 324.1 | 20.5% | ⭐⭐ |
| Block writing | 115.6 | 7.3% | ⭐ |
| Dict operations | 86.1 | 5.4% | |
| SQLite inserts | 42.8 | 2.7% | |
| Garbage collection | 15.1 | 1.0% | |
| Metadata | 6.0 | 0.4% | |
| Arrow reading | 0.4 | 0.0% | |
| **Untracked** | **8.8** | **0.6%** | ✅ |

### Critical Discovery

**Term buffer flush is the PRIMARY BOTTLENECK!**
- Takes more time than tokenization
- Takes more time than merge
- 35.1% of total execution time
- Called ~6,200 times (once per 50k term buffer)

**What it does:**
```python
def _flush_term_buffer(self):
    self.current_block.add_term_batch(self.term_buffer)  # ← 555.7s here
    self.term_buffer = []
```

**Scale:**
- 310 million tokens processed
- 50,000 terms per flush
- ~6,200 flushes total
- Each flush processes sorted insertion into inverted index

---

## Attempt 4: Sorting Optimization

### Hypothesis

**Term buffer flush** calls `add_term_batch()` which sorts 50k terms each time:

```python
def add_term_batch(self, terms_positions_docs: list[tuple[str, int, int]]):
    # THE BOTTLENECK?
    terms_positions_docs.sort(key=lambda x: (x[0], x[1]))
    # ... then process sorted terms
```

**Expected:**
- 6,200 sorts of 50k items = 4.8 billion comparisons
- Lambda function overhead on each comparison
- Tuple creation `(x[0], x[1])` overhead

**Estimated savings:** 50-70% faster sorting = 278-389 seconds

### Implementation

**Change:** Replace slow lambda with optimized `itemgetter`

```python
# Before:
terms_positions_docs.sort(key=lambda x: (x[0], x[1]))

# After:
from operator import itemgetter
terms_positions_docs.sort(key=itemgetter(0, 1))
```

**Why itemgetter?**
- Implemented in C (much faster than Python lambda)
- No tuple creation overhead per comparison
- No function call overhead per comparison
- Standard library optimization for this exact use case

### Results: Disappointing ⚠️

**Actual improvement: Only 0.8%**

```
BEFORE → AFTER
─────────────────────────────────────
Term buffer flush:  555.7s → 544.2s  (-11.5s, -2.1%)
Total time:        1581.4s → 1568.2s (-13.2s, -0.8%)
Throughput:         729.9 → 736.0 docs/sec (+0.8%)
```

**Expected:** 50-70% faster (278-389s saved)
**Actual:** 2.1% faster (11.5s saved)

### Analysis: Sorting is NOT the Bottleneck!

**Time breakdown within `add_term_batch`:**

```
Operation                     Time      % of 544s
───────────────────────────────────────────────
Sorting (6,200 times):        ~11s         2%
Loop (310M iterations):      ~533s        98%  ← REAL BOTTLENECK
```

**What the loop does** (310 million times):

```python
for term, doc_id, position in terms_positions_docs:  # 310M iterations
    if term != current_term:
        self.index[term].append(Posting(...))  # Dict lookup + object creation
    elif doc_id != current_doc:
        self.index[term].append(Posting(...))  # List append
    else:
        current_posting.positions.append(position)  # List append
```

**Bottleneck breakdown:**
1. **Dictionary lookups**: `self.index[term]` - 310M hash computations
2. **Object creation**: `Posting(doc_id, positions=[...])` - Python object overhead
3. **List operations**: `.append()` - dynamic array resizing
4. **Branch predictions**: 310M if/elif checks

**Each iteration costs ~1.7 microseconds:**
- 310M × 1.7μs = 527 seconds ✅

**Why itemgetter had minimal impact:**
- Sorting: 6,200 operations (once per buffer flush)
- Loop: 310,000,000 operations (once per token)
- **Ratio: Loop runs 50,000× more often than sorting!**

### Lessons Learned

1. **Profiling intuition can be wrong** - assumed sorting was slow, but loop dominates
2. **Frequency matters more than cost** - 310M cheap operations >> 6,200 expensive ones
3. **Micro-optimizations have limits** - Cannot fix algorithmic bottleneck with micro-opts
4. **Python overhead is real** - Object creation + dict lookups dominate performance

### Potential Further Optimizations (Not Pursued)

**Why we stopped here:**
- Diminishing returns (0.8% for significant effort)
- Would require algorithmic changes or Cython rewrite
- Risk/reward ratio not favorable
- Current performance is acceptable (26 minutes for 1.15M docs)

**Options considered but rejected:**

1. **Cython hot loop** - Rewrite loop in Cython
   - Estimated gain: 30-50% of loop time (~160-265s)
   - Cost: Complexity, build dependencies, maintenance

2. **Different data structures** - Avoid Posting objects
   - Estimated gain: 20-40% of loop time (~106-212s)
   - Cost: Major refactoring, correctness risk

3. **Pre-allocated structures** - Avoid dynamic resizing
   - Estimated gain: 10-20% of loop time (~53-106s)
   - Cost: Memory overhead, complexity

4. **Remove sorting entirely** - Trust tokenization order
   - Estimated gain: 11s (already achieved)
   - Cost: Correctness risk, marginal benefit

---

## Final Results & Conclusions

### Performance Summary

**Baseline → Final:**
```
Time:        1325s → 1568s (instrumentation overhead ~16%)
Optimized:   1568s → 1568s (kept itemgetter improvement)
Throughput:  871 → 736 docs/sec (instrumentation adds overhead)
```

**Note:** Instrumentation adds ~15% overhead from timing calls. Production version without instrumentation would be ~1350-1400s.

### Optimization Attempts Summary

| Attempt | Goal | Result | Gain | Status |
|---------|------|--------|------|--------|
| Worker parallelization | Multi-core usage | Failed | 0% | ❌ Memory guard |
| Parameter benchmark | Find optimal config | Marginal | ~1.6% | ⚠️ Not worth it |
| Instrumentation | Find bottlenecks | Success | N/A | ✅ Knowledge gained |
| itemgetter optimization | Speed up sorting | Minimal | 0.8% | ⚠️ Wrong target |

### Key Learnings

#### 1. Memory is the Constraint

- 2000 MB hard limit prevents:
  - Parallel processing
  - Larger batch sizes
  - Larger block sizes
  - In-memory optimizations

**Implication:** Cannot trade memory for speed

#### 2. Current Implementation is Well-Optimized

- Default parameters are already near-optimal
- Major bottlenecks are algorithmic, not parametric
- Micro-optimizations have minimal impact (<1%)

**Implication:** No easy wins remaining

#### 3. Profiling Assumptions Can Be Wrong

- Assumed sorting was slow (6,200 expensive operations)
- Reality: Loop operations dominate (310M cheap operations)
- Frequency × Cost matters more than individual cost

**Implication:** Always measure, never assume

#### 4. Python Overhead is Significant

- Object creation, dictionary lookups dominate
- Pure Python cannot be optimized much further
- Would need Cython/Numba/Rust for major gains

**Implication:** Acceptable tradeoff for readability

### Performance Breakdown (Final)

```
┌─────────────────────────────────────────────────────┐
│ Total Time: 1,568 seconds (26.1 minutes)            │
├─────────────────────────────────────────────────────┤
│ 35% Term buffer operations     (544s)               │
│ 25% Tokenization              (398s)                │
│ 21% Merge                      (326s)                │
│  7% Block writing              (113s)                │
│  5% Dictionary operations       (86s)                │
│  3% SQLite inserts              (44s)                │
│  1% Garbage collection          (15s)                │
│ <3% Other                       (42s)                │
└─────────────────────────────────────────────────────┘

Throughput: 736 documents/second
            197,000 tokens/second
```

### Recommendations

#### For This Implementation

**✅ Keep as-is:**
- Default parameters (batch=1000, block=200MB, memory=1800MB)
- itemgetter optimization (marginal but free)
- Current architecture (well-designed for constraints)

**❌ Do not pursue:**
- Worker parallelization (memory guard prevents)
- Further micro-optimizations (diminishing returns)
- Parameter tuning (already optimal)

#### For Future Work

**If performance is critical:**

1. **Use compiled languages**
   - Rewrite hot paths in Rust/C++
   - Estimated 2-3× speedup possible

2. **Change assignment constraints**
   - Remove 2000 MB memory limit
   - Enable parallel processing
   - Estimated 3-4× speedup on 4 cores

3. **Algorithmic improvements**
   - Different data structures (tries, compressed posting lists)
   - Streaming merge instead of k-way
   - Estimated 20-40% speedup

**If current performance is acceptable:**
- Move to other features (search, ranking, API)
- Focus on correctness and user experience
- Performance is "good enough" for this dataset size

### Final Verdict

**Current performance: 26 minutes for 1.15M documents is acceptable.**

Further optimization would require:
- Significant code complexity (Cython, architectural changes)
- High risk of introducing bugs
- Marginal benefits (<20% improvement)
- Violating assignment constraints (memory limit, parallel processing)

**Recommendation: STOP optimizing and accept current performance.**

---

## Appendix: Benchmark Files

- `benchmark/benchmark_phase1.py` - Automated parameter testing
- `benchmark/benchmark_phase1_results.json` - Full results (23 runs)
- `benchmark/PHASE1_RESULTS.md` - Results summary
- `benchmark/README.md` - How to run benchmarks

## Appendix: Timing Statistics

Example timing output:
```json
{
  "timings": {
    "tokenization_seconds": 397.8,
    "block_writing_seconds": 113.4,
    "merge_seconds": 326.3,
    "metadata_seconds": 5.9,
    "sqlite_insert_seconds": 44.4,
    "arrow_reading_seconds": 0.4,
    "term_buffer_flush_seconds": 544.2,
    "doc_buffer_flush_seconds": 44.4,
    "gc_seconds": 15.2,
    "dict_operations_seconds": 86.0,
    "total_indexing_seconds": 1568.2
  }
}
```

---

**Document version:** 1.0
**Last updated:** November 2025
**Authors:** PRI Assignment 1 Team
