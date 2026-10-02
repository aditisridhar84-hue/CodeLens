"""CodeLens package alias to codetrace modules."""

from codetrace import chunker, graph, intent, retrievers, pipeline, versions, git_tracker, evaluate, api

from codetrace.chunker import CodeChunk, chunk_source_code, chunk_python_source, chunk_js_source, chunk_repo
from codetrace.graph import CodeGraph
from codetrace.intent import QueryAnalysis, analyze_query
from codetrace.retrievers import DenseRetriever, BM25Retriever, tokenize, min_max_normalize, reciprocal_rank_fusion
from codetrace.pipeline import Pipeline, SearchResult
from codetrace.versions import HistoryIndex, build_history, link_versions, save_sqlite, load_sqlite
from codetrace.git_tracker import GitTracker

__all__ = [
    "CodeChunk",
    "chunk_source_code",
    "chunk_python_source",
    "chunk_js_source",
    "chunk_repo",
    "CodeGraph",
    "QueryAnalysis",
    "analyze_query",
    "DenseRetriever",
    "BM25Retriever",
    "tokenize",
    "min_max_normalize",
    "reciprocal_rank_fusion",
    "Pipeline",
    "SearchResult",
    "HistoryIndex",
    "build_history",
    "link_versions",
    "save_sqlite",
    "load_sqlite",
    "GitTracker",
]
