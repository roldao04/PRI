# Phase 1 Benchmark Results

**Date**: 2025-11-07/08
**Dataset**: 1,154,228 documents (ptwiki-articles-with-redirects.arrow)
**Total runs**: 23 (18 successful, 5 memory guard kills)

## Summary

Tested 4 parameters independently to identify impact on indexing speed.

## Results by Parameter

### 1. Indexer Batch Size ⭐ BIGGEST IMPACT

| Value | Time (s) | Throughput (docs/s) | Status | Improvement |
|-------|----------|---------------------|--------|-------------|
| 250   | 1683     | 685.8               | ✅     | baseline    |
| 500   | 1630     | 708.1               | ✅     | +3.2%       |
| **1000** | **1575** | **732.9**       | ✅     | **+6.4%**   |
| 1500  | -        | -                   | ❌ OOM | -           |
| 2000  | -        | -                   | ❌ OOM | -           |
| 3000  | -        | -                   | ❌ OOM | -           |

**Finding**: Batch size 1000 is optimal. Larger values trigger memory guard kills.

### 2. Block Size (MB)

| Value | Time (s) | Throughput (docs/s) | Blocks Created | Status | Improvement |
|-------|----------|---------------------|----------------|--------|-------------|
| 100   | 1579     | 730.9               | 109            | ✅     | baseline    |
| 150   | 1575     | 732.9               | 100            | ✅     | +0.25%      |
| 200   | 1574     | 733.2               | 83             | ✅     | +0.32%      |
| **250** | **1572** | **734.2**       | **67**         | ✅     | **+0.45%**  |
| 300   | -        | -                   | -              | ❌ OOM | -           |
| 400   | -        | -                   | -              | ❌ OOM | -           |

**Finding**: Larger blocks = fewer blocks created = slightly faster merge. 250MB is optimal.

### 3. Memory Limit (MB) 🔍 COUNTER-INTUITIVE

| Value | Time (s) | Throughput (docs/s) | Status | Improvement |
|-------|----------|---------------------|--------|-------------|
| **1200** | **1569** | **735.5**       | ✅     | **+0.69%**  |
| 1400  | 1571     | 734.6               | ✅     | +0.57%      |
| 1600  | 1573     | 733.8               | ✅     | +0.38%      |
| 1800  | 1574     | 733.5               | ✅     | baseline    |
| 2000  | 1580     | 730.7               | ✅     | -0.38%      |

**Finding**: LOWER memory limit performs BETTER. Forces more frequent block writes, prevents memory buildup overhead.

### 4. Tokenizer Batch Size

| Value | Time (s) | Throughput (docs/s) | Status | Improvement |
|-------|----------|---------------------|--------|-------------|
| **1000** | **1567** | **736.8**       | ✅     | **+0.50%**  |
| 2500  | 1571     | 734.7               | ✅     | +0.25%      |
| 5000  | 1574     | 733.2               | ✅     | baseline    |
| 7500  | 1574     | 733.2               | ✅     | 0%          |
| 10000 | 1574     | 733.1               | ✅     | 0%          |
| 15000 | 1580     | 730.7               | ✅     | -0.32%      |

**Finding**: Smaller tokenizer batches slightly faster. Minimal impact overall.

## Optimal Configuration

```python
indexer_batch_size: 1000      # keep default ✅
block_size_mb: 250            # increase from 200 ⬆️
memory_limit_mb: 1200         # decrease from 1800 ⬇️
tokenizer_batch_size: 1000    # decrease from 5000 ⬇️
```

**Expected performance**: ~1566s (26.1 min)
**Current baseline**: 1574s (26.2 min)
**Estimated improvement**: ~0.5% (8 seconds)

## Key Insights

1. **Current defaults are already well-optimized** - improvements are marginal
2. **Memory constraints are the bottleneck** - larger batch/block sizes cause OOM
3. **Lower memory limits perform better** - counter-intuitive but forces efficient block management
4. **Indexer batch size has the biggest impact** - but limited by memory guard

## Memory Guard Failures

5 configurations failed with memory guard kills:
- `indexer_batch_size >= 1500` (all failed)
- `block_size_mb >= 300` (all failed)

These trigger when actual memory usage exceeds 2000MB hard limit.

## Recommendations

1. **Keep indexer_batch_size=1000** (optimal and safe)
2. **Test block_size=250 with memory_limit=1200** together (may conflict)
3. **Investigate memory usage patterns** to understand OOM triggers
4. **Consider adjusting memory guard threshold** if safe to do so

---

**Full results**: `benchmark_phase1_results.json`
