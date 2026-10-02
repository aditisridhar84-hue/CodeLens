"""Reproducible evaluation runner for CodeLens / CoIR AppsRetrieval.

Usage:
    python evaluation/run_eval.py [--repo <path>] [--output <csv_path>]
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from codetrace.chunker import chunk_source_code
from codetrace.evaluate import DEFAULT_BENCHMARK_QUERIES, evaluate_pipeline
from codetrace.pipeline import Pipeline
from codetrace.versions import build_history


def run():
    parser = argparse.ArgumentParser(description="Run CodeLens IR Evaluation")
    parser.add_argument("--repo", default=".", help="Repository path to index")
    parser.add_argument("--limit", type=int, default=50, help="Max commits to evaluate")
    parser.add_argument("--benchmark", default=str(PROJECT_ROOT / "evaluation" / "benchmark_data.json"))
    parser.add_argument("--output", default="ablation_results.csv", help="CSV output path")
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    if benchmark_path.exists():
        with open(benchmark_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            queries = data.get("queries", DEFAULT_BENCHMARK_QUERIES)
    else:
        queries = DEFAULT_BENCHMARK_QUERIES

    print(f"Indexing repository at '{args.repo}'...")
    t0 = time.perf_counter()
    hist = build_history(args.repo, max_commits=args.limit)

    # If repo has no chunks, use synthetic sample files
    if not hist.chunks:
        from codetrace.evaluate import _create_eval_corpus
        hist.chunks = _create_eval_corpus()

    index_time = time.perf_counter() - t0
    print(f"Indexed {len(hist.chunks)} chunks across {len(hist.commits)} commits in {index_time:.2f}s")

    print(f"Running evaluation on {len(queries)} queries across 4 pipeline modes...")
    pipe = Pipeline(hist.chunks)
    df = evaluate_pipeline(pipe, queries)

    df.to_csv(args.output, index=False)
    print("\n" + "=" * 60)
    print("CodeLens Evaluation Results:")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    print(f"Commit-to-index time: {index_time:.2f}s")
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    run()
