#!/usr/bin/env python3
"""Phase 1 Benchmark: Test individual parameter impact on indexing performance.

This script tests each parameter independently while keeping others at default values:
- Indexer batch_size: [250, 500, 1000, 1500, 2000, 3000]
- Block size (MB): [100, 150, 200, 250, 300, 400]
- Memory limit (MB): [1200, 1400, 1600, 1800, 2000]
- Tokenizer batch_size: [1000, 2500, 5000, 7500, 10000, 15000]

Total: 23 runs, estimated ~8.5 hours with full dataset
"""

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATASET_PATH = BASE_DIR / "ptwiki-articles-with-redirects.arrow"
OUTPUT_DIR = BASE_DIR / "benchmark"
RESULTS_FILE = OUTPUT_DIR / "benchmark_phase1_results.json"
CLI_SCRIPT = BASE_DIR / "src" / "indexa" / "entrypoints" / "cli.py"

# Default values (matching current implementation)
DEFAULTS = {
    "indexer_batch_size": 1000,
    "block_size_mb": 200,
    "memory_limit_mb": 1800,
    "tokenizer_batch_size": 5000,
    "min_term_frequency": 1,
}

# Parameter test ranges
TEST_CONFIGS = {
    "indexer_batch_size": [250, 500, 1000, 1500, 2000, 3000],
    "block_size_mb": [100, 150, 200, 250, 300, 400],
    "memory_limit_mb": [1200, 1400, 1600, 1800, 2000],
    "tokenizer_batch_size": [1000, 2500, 5000, 7500, 10000, 15000],
}


def load_existing_results() -> List[Dict[str, Any]]:
    """Load existing benchmark results if they exist."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_results(results: List[Dict[str, Any]]) -> None:
    """Save benchmark results to JSON file."""
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to {RESULTS_FILE}")


def generate_test_configurations() -> List[Dict[str, Any]]:
    """Generate all test configurations for Phase 1."""
    configs = []

    for param_name, test_values in TEST_CONFIGS.items():
        for value in test_values:
            # Start with defaults
            config = DEFAULTS.copy()
            # Vary this parameter
            config[param_name.replace("_", "_")] = value

            configs.append(
                {
                    "varied_parameter": param_name,
                    "varied_value": value,
                    "config": config,
                }
            )

    return configs


def has_already_run(results: List[Dict[str, Any]], config: Dict[str, Any]) -> bool:
    """Check if this configuration has already been tested successfully."""
    for result in results:
        if (
            result.get("varied_parameter") == config["varied_parameter"]
            and result.get("varied_value") == config["varied_value"]
            and result.get("status") == "success"
        ):
            return True
    return False


def cleanup_index_dir(index_dir: Path) -> None:
    """Remove index directory to save disk space."""
    if index_dir.exists():
        try:
            shutil.rmtree(index_dir)
            print(f"  Cleaned up {index_dir.name}")
        except Exception as e:
            print(f"  Warning: Could not clean up {index_dir}: {e}")


def run_indexer(config: Dict[str, Any], run_number: int, total_runs: int) -> Dict[str, Any]:
    """Run the indexer with given configuration and collect results.

    Returns a dictionary with status, timing, and error information.
    """
    # Create unique output directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_name = config["varied_parameter"]
    param_value = config["varied_value"]
    index_dir = OUTPUT_DIR / f"run_{timestamp}_{param_name}_{param_value}"

    print(f"\n{'='*80}")
    print(f"Run {run_number}/{total_runs}: Testing {param_name} = {param_value}")
    print(f"{'='*80}")
    print("Configuration:")
    for key, value in config["config"].items():
        marker = "◄" if key == param_name else ""
        print(f"  {key}: {value} {marker}")
    print()

    # Build command
    # Use the project's CLI entry point (works with uv/pip installed packages)
    venv_python = BASE_DIR / ".venv" / "bin" / "python"
    python_executable = str(venv_python) if venv_python.exists() else sys.executable

    cmd = [
        python_executable,
        "-m",
        "indexa.entrypoints.cli",
        str(DATASET_PATH),
        "-o",
        str(index_dir),
        "--batch-size",
        str(config["config"]["indexer_batch_size"]),
        "--block-size",
        str(config["config"]["block_size_mb"]),
        "--memory-limit",
        str(config["config"]["memory_limit_mb"]),
        "--min-term-freq",
        str(config["config"]["min_term_frequency"]),
    ]

    # Handle tokenizer batch_size via config file
    tokenizer_config_file = None
    if param_name == "tokenizer_batch_size":
        # Create temporary tokenizer config file
        tokenizer_config_file = OUTPUT_DIR / f"tokenizer_config_{timestamp}.json"
        tokenizer_config = {
            "use_stemming": True,
            "use_stopwords": True,
            "min_token_length": 3,
            "max_token_length": 50,
            "batch_size": config["config"]["tokenizer_batch_size"],
            "keep_numbers": False,
        }
        with open(tokenizer_config_file, "w") as f:
            json.dump(tokenizer_config, f, indent=2)
        cmd.extend(["--tokenizer-config", str(tokenizer_config_file)])
        print(f"  Using tokenizer config file: {tokenizer_config_file}")

    result = {
        "run_number": run_number,
        "timestamp": timestamp,
        "varied_parameter": param_name,
        "varied_value": param_value,
        "configuration": config["config"],
        "status": "unknown",
        "error_message": None,
    }

    start_time = time.time()

    try:
        print(f"Starting indexer at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")

        # Run indexer (must run from base directory for module imports to work)
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout per run
            cwd=BASE_DIR,  # Run from base directory
        )

        elapsed = time.time() - start_time

        # Check if successful
        if process.returncode == 0:
            # Try to read timing stats
            timing_stats_file = index_dir / "timing_stats.json"

            if timing_stats_file.exists():
                with open(timing_stats_file) as f:
                    timing_stats = json.load(f)

                result["status"] = "success"
                result["timings"] = timing_stats["timings"]
                result["throughput"] = timing_stats["throughput"]
                result["statistics"] = timing_stats["statistics"]
                result["elapsed_seconds"] = elapsed

                print(f"✓ Success! Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
                total_time = timing_stats["timings"]["total_indexing_seconds"]
                print(f"  Total indexing time: {total_time:.1f}s")
                throughput = timing_stats["throughput"]["documents_per_second"]
                print(f"  Throughput: {throughput:.1f} docs/sec")
                print(f"  Blocks created: {timing_stats['statistics']['blocks_created']}")
            else:
                result["status"] = "error"
                result["error_message"] = "Timing stats file not found"
                print("✗ Error: timing_stats.json not found")

        elif process.returncode == -9:
            # Killed by OS (likely OOM)
            result["status"] = "memory_guard_triggered"
            result["error_message"] = "Process killed (likely OOM/memory guard)"
            print("✗ Memory guard triggered! Process was killed (OOM)")

        else:
            # Other error
            result["status"] = "crashed"
            result["error_message"] = f"Exit code {process.returncode}"
            result["stderr"] = process.stderr[-1000:] if process.stderr else None  # Last 1000 chars
            print(f"✗ Crashed with exit code {process.returncode}")
            if process.stderr:
                print(f"  Error: {process.stderr[-500:]}")

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error_message"] = "Process timed out after 2 hours"
        print("✗ Timeout! Process exceeded 2 hour limit")

    except Exception as e:
        result["status"] = "error"
        result["error_message"] = str(e)
        print(f"✗ Error: {e}")

    finally:
        # Cleanup index directory to save disk space
        cleanup_index_dir(index_dir)

        # Cleanup tokenizer config file if created
        if tokenizer_config_file and tokenizer_config_file.exists():
            try:
                tokenizer_config_file.unlink()
            except Exception as e:
                print(f"  Warning: Could not clean up tokenizer config: {e}")

    return result


def print_progress_summary(results: List[Dict[str, Any]], total_runs: int) -> None:
    """Print summary of completed runs and estimated time remaining."""
    completed = len(results)
    successful = sum(1 for r in results if r["status"] == "success")
    failed = completed - successful

    # Calculate average time per run
    successful_times = [
        r.get("elapsed_seconds", 0)
        for r in results
        if r["status"] == "success" and "elapsed_seconds" in r
    ]

    if successful_times:
        avg_time = sum(successful_times) / len(successful_times)
        remaining_runs = total_runs - completed
        eta_seconds = avg_time * remaining_runs
        eta = timedelta(seconds=int(eta_seconds))

        print(f"\n{'='*80}")
        print("Progress Summary:")
        print(f"  Completed: {completed}/{total_runs} runs ({completed/total_runs*100:.1f}%)")
        print(f"  Successful: {successful}, Failed: {failed}")
        print(f"  Average time per run: {avg_time/60:.1f} minutes")
        print(f"  Estimated time remaining: {eta}")
        print(f"  Estimated completion: {datetime.now() + eta:%Y-%m-%d %H:%M}")
        print(f"{'='*80}\n")


def print_final_summary(results: List[Dict[str, Any]]) -> None:
    """Print final summary of all benchmark runs."""
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE!")
    print(f"{'='*80}")

    by_status = {}
    for r in results:
        status = r["status"]
        by_status[status] = by_status.get(status, 0) + 1

    print("\nResults by status:")
    for status, count in sorted(by_status.items()):
        print(f"  {status}: {count}")

    # Show any failed runs
    failed = [r for r in results if r["status"] != "success"]
    if failed:
        print("\nFailed runs:")
        for r in failed:
            param = r["varied_parameter"]
            value = r["varied_value"]
            status = r["status"]
            error = r.get("error_message", "N/A")
            print(f"  - {param}={value}: {status} - {error}")

    # Find fastest configuration
    successful = [r for r in results if r["status"] == "success"]
    if successful:
        fastest = min(successful, key=lambda r: r["timings"]["total_indexing_seconds"])
        print("\nFastest configuration:")
        print(f"  Parameter: {fastest['varied_parameter']} = {fastest['varied_value']}")
        print(f"  Total time: {fastest['timings']['total_indexing_seconds']:.1f}s")
        print(f"  Throughput: {fastest['throughput']['documents_per_second']:.1f} docs/sec")

    print(f"\nResults saved to: {RESULTS_FILE}")
    print(f"{'='*80}\n")


def main():
    """Main benchmark execution."""
    print(f"\n{'='*80}")
    print("PHASE 1 BENCHMARK: Individual Parameter Testing")
    print(f"{'='*80}\n")

    # Validate dataset exists
    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        sys.exit(1)

    # Validate CLI exists
    if not CLI_SCRIPT.exists():
        print(f"Error: CLI script not found at {CLI_SCRIPT}")
        sys.exit(1)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing results (for resume capability)
    results = load_existing_results()
    if results:
        print(f"Found {len(results)} existing results, resuming from there...\n")

    # Generate test configurations
    configs = generate_test_configurations()
    total_runs = len(configs)

    print("Configuration:")
    print(f"  Dataset: {DATASET_PATH}")
    print(f"  Total test runs: {total_runs}")
    print("  Parameters to test:")
    for param_name, values in TEST_CONFIGS.items():
        print(f"    - {param_name}: {len(values)} values {values}")
    print(f"  Results file: {RESULTS_FILE}")
    print("\nEstimated total time: ~8.5 hours")
    print("\nPress Ctrl+C to stop (progress will be saved)\n")

    time.sleep(3)  # Give user time to read

    # Run benchmarks
    try:
        for i, config in enumerate(configs, 1):
            # Skip if already run
            if has_already_run(results, config):
                param = config["varied_parameter"]
                value = config["varied_value"]
                print(f"Skipping run {i}/{total_runs}: {param}={value} (already completed)")
                continue

            # Run benchmark
            result = run_indexer(config, i, total_runs)

            # Save result
            results.append(result)
            save_results(results)

            # Print progress
            if i < total_runs:
                print_progress_summary(results, total_runs)
                print("Waiting 5 seconds before next run...\n")
                time.sleep(5)

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user!")
        print(f"Progress has been saved. Run again to resume from run {len(results) + 1}")
        save_results(results)
        sys.exit(0)

    # Print final summary
    print_final_summary(results)


if __name__ == "__main__":
    main()
