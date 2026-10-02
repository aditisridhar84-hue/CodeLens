# CodeTrace AI
Intent-aware hybrid code retrieval (BM25 + embeddings + RRF + rerank + confidence pass) with git-version tracking.

## Install
    python -m venv .venv && .venv\Scripts\activate   # or source .venv/bin/activate on Unix
    pip install -r requirements.txt
    pip install -e .

## Run
    streamlit run app.py
    python -m codetrace --help
    python -m codetrace index --repo . --json
    python -m codetrace search --repo . --query "How is input normalized before the main function?" --json
    python -m codetrace eval --repo . --limit 10 --json
    python -m codetrace serve --host 127.0.0.1 --port 8000

## Benchmark (ablation A-D)
    python -m codetrace.evaluate --limit 300 --max-docs 5000
Results go to `ablation_results.csv` and appear in the app's Benchmark tab.

## Layout
- `codetrace/chunker.py` function chunks · `retrievers.py` BM25/dense/RRF · `intent.py` query decomposition
- `codetrace/pipeline.py` modes A-D · `versions.py` git history + version links + SQLite · `evaluate.py` NDCG/MRR
