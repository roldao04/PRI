#!/usr/bin/env python3
"""Quick test of benchmark script with a small dataset limit.

Use this to verify the benchmark works before running overnight.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
BENCHMARK_SCRIPT = BASE_DIR / "benchmark" / "benchmark_phase1.py"

print("Testing benchmark script with max_docs limit...")
print("This will run a quick test with only 1000 documents\n")

# Create a modified version that tests with small dataset
test_code = """
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Monkey patch to use small dataset
original_file = open(__file__.replace('_test.py', '_phase1.py')).read()
exec(original_file.replace('max_docs: int | None = None', 'max_docs: int | None = 1000'))
"""

# Simpler approach: just run a single configuration manually
print("Running single test configuration:")
print("  batch_size=1000, block_size=200, memory_limit=1800\n")

venv_python = BASE_DIR / ".venv" / "bin" / "python"
python_executable = str(venv_python) if venv_python.exists() else sys.executable

dataset_path = BASE_DIR / "ptwiki-articles-with-redirects.arrow"
output_dir = BASE_DIR / "benchmark" / "test_run"

cmd = [
    python_executable,
    "-m",
    "indexa.entrypoints.cli",
    str(dataset_path),
    "-o",
    str(output_dir),
    "--batch-size",
    "1000",
    "--block-size",
    "200",
    "--memory-limit",
    "1800",
    "--min-term-freq",
    "1",
    "--max-docs",
    "1000",  # Small test
]

print(f"Command: {' '.join(cmd)}\n")

try:
    result = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        timeout=120,  # 2 minute timeout for small test
    )

    if result.returncode == 0:
        print("✓ Test SUCCESSFUL!")

        # Check for timing stats
        timing_file = output_dir / "timing_stats.json"
        if timing_file.exists():
            with open(timing_file) as f:
                stats = json.load(f)

            print("\nTiming stats:")
            print(f"  Documents indexed: {stats['statistics']['documents_indexed']}")
            print(f"  Total time: {stats['timings']['total_indexing_seconds']:.1f}s")
            print(f"  Throughput: {stats['throughput']['documents_per_second']:.1f} docs/sec")
            print("\n✓ Benchmark script should work correctly!")
            print("\nTo run the full benchmark overnight:")
            print("  python3 benchmark/benchmark_phase1.py")
        else:
            print("✗ Warning: timing_stats.json not found")
    else:
        print(f"✗ Test FAILED with exit code {result.returncode}")
        if result.stderr:
            print(f"\nError output:\n{result.stderr[-500:]}")

except subprocess.TimeoutExpired:
    print("✗ Test TIMEOUT (took longer than 2 minutes)")
except Exception as e:
    print(f"✗ Test ERROR: {e}")

# Cleanup
if output_dir.exists():
    shutil.rmtree(output_dir)
    print(f"\nCleaned up test directory: {output_dir}")
