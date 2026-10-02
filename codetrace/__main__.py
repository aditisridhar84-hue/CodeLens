"""Command-line entrypoint for the CodeLens / CodeTrace package."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence


def _cmd_index(args) -> int:
    from .versions import build_history

    hist = build_history(args.repo, max_commits=args.max_commits)
    payload = {
        "repo": args.repo,
        "chunks": len(hist.chunks),
        "commits": len(hist.commits),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"CodeLens: indexed {len(hist.chunks)} chunks across {len(hist.commits)} commits.")
    return 0


def _cmd_search(args) -> int:
    from .pipeline import Pipeline
    from .versions import build_history

    hist = build_history(args.repo, max_commits=args.max_commits)
    if not hist.chunks:
        raise SystemExit("No code chunks found in repository.")

    pipe = Pipeline(hist.chunks)
    results = pipe.search(args.query, mode=args.mode, k=args.k)
    payload = {
        "query": args.query,
        "mode": args.mode,
        "results": [
            {
                "rank": i,
                "symbol": (result.chunk.name or "<anonymous>"),
                "path": result.chunk.path,
                "language": result.chunk.language,
                "start": result.chunk.start,
                "end": result.chunk.end,
                "score": round(result.score, 4),
                "commit": result.chunk.commit,
                "code": result.chunk.code,
            }
            for i, result in enumerate(results, 1)
        ],
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"CodeLens: found {len(results)} matches for '{args.query}'")
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            print(f"{i}. {chunk.name or '<anonymous>'} :: {chunk.path}:{chunk.start}-{chunk.end} :: score={result.score:.3f}")
    return 0


def _cmd_eval(args) -> int:
    from .evaluate import run_evaluation

    print(f"CodeLens evaluation starting for repository '{args.repo}'")
    df = run_evaluation(repo_path=args.repo, output_csv=args.output, limit=args.limit)
    if args.json:
        print(json.dumps(df.to_dict(orient="records"), indent=2))
    else:
        print(df.to_string(index=False))
    return 0


def _cmd_serve(args) -> int:
    import uvicorn

    print(f"CodeLens API starting at http://{args.host}:{args.port}")
    uvicorn.run("codetrace.api:app", host=args.host, port=args.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codetrace",
        description="CodeLens / CodeTrace AI: git-aware code retrieval and benchmarking",
    )
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="build a repository history index")
    index_parser.add_argument("--repo", default=".", help="repository path")
    index_parser.add_argument("--max-commits", type=int, default=50, help="maximum commits to index")
    index_parser.add_argument("--json", action="store_true", help="emit JSON output")
    index_parser.set_defaults(func=_cmd_index)

    search_parser = subparsers.add_parser("search", help="run a code search query")
    search_parser.add_argument("--repo", default=".", help="repository path")
    search_parser.add_argument("--query", required=True, help="natural language query")
    search_parser.add_argument("--mode", choices=["dense", "hybrid", "rerank", "full"], default="full")
    search_parser.add_argument("--k", type=int, default=5)
    search_parser.add_argument("--max-commits", type=int, default=50)
    search_parser.add_argument("--json", action="store_true", help="emit JSON output")
    search_parser.set_defaults(func=_cmd_search)

    eval_parser = subparsers.add_parser("eval", help="run benchmark evaluation")
    eval_parser.add_argument("--repo", default=".", help="repository path")
    eval_parser.add_argument("--limit", type=int, default=50)
    eval_parser.add_argument("--output", default="ablation_results.csv")
    eval_parser.add_argument("--json", action="store_true", help="emit JSON output")
    eval_parser.set_defaults(func=_cmd_eval)

    serve_parser = subparsers.add_parser("serve", help="run the FastAPI service")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(func=_cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
