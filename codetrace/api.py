"""FastAPI REST API Service for CodeLens.

Implements all required CodeLens endpoints:
- POST /repositories/connect
- GET  /repositories
- POST /repositories/{id}/index
- GET  /repositories/{id}/status
- POST /search
- GET  /repositories/{id}/commits
- GET  /repositories/{id}/versions
- GET  /chunks/{id}
- GET  /chunks/{id}/evolution
- POST /webhooks/github
- POST /repositories/{id}/sync
- GET  /evaluation
"""

from __future__ import annotations

import difflib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .chunker import CodeChunk
from .evaluate import DEFAULT_BENCHMARK_QUERIES, evaluate_pipeline
from .git_tracker import GitTracker
from .pipeline import Pipeline, SearchResult
from .versions import HistoryIndex, build_history, link_versions, load_sqlite, save_sqlite

app = FastAPI(
    title="CodeLens API",
    description="Git-Aware Multi-Stage Code Retrieval Engine",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

git_tracker = GitTracker()

# In-memory storage for repositories and active pipelines
REPOSITORIES: Dict[str, Dict[str, Any]] = {}
PIPELINES: Dict[str, Pipeline] = {}
HISTORIES: Dict[str, HistoryIndex] = {}
LINKS: Dict[str, List[Dict[str, Any]]] = {}


# --- Request & Response Models ---
class ConnectRepoRequest(BaseModel):
    url: str = Field(..., description="Local directory path or GitHub URL")
    branch: Optional[str] = Field(None, description="Optional Git branch name")
    max_commits: int = Field(50, description="Max commits to index into history")


class SearchRequest(BaseModel):
    repository_id: str
    query: str
    mode: str = Field("full", description="dense, hybrid, rerank, or full")
    k: int = Field(10, description="Number of results to return")
    commit_sha: Optional[str] = Field(None, description="Filter as of specific commit SHA")
    language: Optional[str] = Field(None, description="Filter by programming language")


class SearchResultItem(BaseModel):
    id: str
    symbol: str
    path: str
    language: str
    start_line: int
    end_line: int
    score: float
    commit: str
    code: str
    doc: str
    calls: List[str]
    called_by: List[str]
    parent_class: Optional[str]
    evidence: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    mode: str
    results_count: int
    latency_ms: float
    results: List[SearchResultItem]


# --- Helper Functions ---
def get_or_create_repo(repo_id: str) -> Dict[str, Any]:
    if repo_id not in REPOSITORIES:
        raise HTTPException(status_code=404, detail=f"Repository '{repo_id}' not found")
    return REPOSITORIES[repo_id]


# --- Endpoints ---
@app.get("/")
def root():
    return {
        "engine": "CodeLens",
        "status": "online",
        "version": "1.0.0",
        "description": "Git-Aware Multi-Stage Code Retrieval Engine",
    }


@app.post("/repositories/connect")
def connect_repository(req: ConnectRepoRequest):
    """Connect a local repository or clone/fetch a GitHub repository."""
    t0 = time.perf_counter()
    try:
        local_path, current_sha, repo_obj = git_tracker.ingest_repository(
            req.url, branch=req.branch, max_commits=req.max_commits
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to ingest repository: {str(e)}")

    repo_id = Path(local_path).name.lower()
    hist = build_history(local_path, max_commits=req.max_commits, repo_url=req.url)

    if not hist.chunks:
        raise HTTPException(status_code=400, detail="No code files found in repository.")

    pipe = Pipeline(hist.chunks)
    links = link_versions(hist)
    save_sqlite(hist, links, db_path=f"data/codelens_{repo_id}.db")

    elapsed = time.perf_counter() - t0

    REPOSITORIES[repo_id] = {
        "id": repo_id,
        "url": req.url,
        "local_path": local_path,
        "branch": req.branch or "default",
        "current_sha": current_sha,
        "status": "READY",
        "files_count": len({c.path for c in hist.chunks}),
        "chunks_count": len(hist.chunks),
        "index_time": round(elapsed, 2),
        "last_sync": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    PIPELINES[repo_id] = pipe
    HISTORIES[repo_id] = hist
    LINKS[repo_id] = links

    return {
        "message": "Repository connected and indexed successfully",
        "repository": REPOSITORIES[repo_id],
    }


@app.get("/repositories")
def list_repositories():
    """List all connected repositories."""
    return list(REPOSITORIES.values())


@app.post("/repositories/{repo_id}/index")
def index_repository(repo_id: str):
    """Trigger re-indexing or incremental sync on a repository."""
    repo = get_or_create_repo(repo_id)
    hist = HISTORIES[repo_id]
    pipe = PIPELINES[repo_id]

    hist, pipe, stats = git_tracker.incremental_index(hist, pipe, repo["local_path"])
    HISTORIES[repo_id] = hist
    PIPELINES[repo_id] = pipe
    repo["current_sha"] = hist.latest_commit["sha"]
    repo["last_sync"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {"status": "success", "stats": stats}


@app.get("/repositories/{repo_id}/status")
def get_repository_status(repo_id: str):
    """Get current indexing status, commit SHA, file count, and chunk stats."""
    repo = get_or_create_repo(repo_id)
    hist = HISTORIES[repo_id]
    return {
        "repository": repo,
        "latest_commit": hist.latest_commit,
        "commits_count": len(hist.commits),
        "chunks_count": len(hist.chunks),
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Execute multi-signal code search across the repository."""
    repo = get_or_create_repo(req.repository_id)
    pipe = PIPELINES[req.repository_id]
    hist = HISTORIES[req.repository_id]

    t0 = time.perf_counter()
    commit_filter = req.commit_sha
    results = pipe.search(
        req.query,
        mode=req.mode,
        k=req.k,
        commit_filter=commit_filter,
        language_filter=req.language,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    items = []
    for r in results:
        c = r.chunk
        items.append(
            SearchResultItem(
                id=c.id,
                symbol=c.name or "anonymous",
                path=c.path,
                language=c.language,
                start_line=c.start,
                end_line=c.end,
                score=round(r.score, 4),
                commit=c.commit,
                code=c.code,
                doc=c.doc,
                calls=c.calls,
                called_by=c.called_by,
                parent_class=c.parent_class,
                evidence=r.evidence,
            )
        )

    return SearchResponse(
        query=req.query,
        mode=req.mode,
        results_count=len(items),
        latency_ms=round(latency_ms, 1),
        results=items,
    )


@app.get("/repositories/{repo_id}/commits")
def get_repository_commits(repo_id: str):
    """Get list of commits indexed for the repository."""
    get_or_create_repo(repo_id)
    hist = HISTORIES[repo_id]
    return {"commits": hist.commits}


@app.get("/repositories/{repo_id}/versions")
def get_repository_versions(repo_id: str):
    """Get detected version links and code evolution history."""
    get_or_create_repo(repo_id)
    links = LINKS.get(repo_id, [])
    return {"links_count": len(links), "links": links}


@app.get("/chunks/{chunk_id}")
def get_chunk_details(chunk_id: str):
    """Get full details of a specific code chunk including graph neighborhood."""
    for repo_id, pipe in PIPELINES.items():
        chunk = pipe.chunk_by_id.get(chunk_id)
        if chunk:
            neighborhood = pipe.code_graph.get_neighborhood(chunk_id)
            return {
                "chunk": {
                    "id": chunk.id,
                    "symbol": chunk.name,
                    "path": chunk.path,
                    "language": chunk.language,
                    "start": chunk.start,
                    "end": chunk.end,
                    "commit": chunk.commit,
                    "code": chunk.code,
                    "doc": chunk.doc,
                    "imports": chunk.imports,
                    "calls": chunk.calls,
                    "called_by": chunk.called_by,
                    "parent_class": chunk.parent_class,
                    "keywords": chunk.keywords,
                },
                "neighborhood": neighborhood,
            }
    raise HTTPException(status_code=404, detail="Chunk not found")


@app.get("/chunks/{chunk_id}/evolution")
def get_chunk_evolution(chunk_id: str):
    """Get evolution lineage across commits for a chunk's symbol."""
    target_chunk = None
    target_repo_id = None
    for r_id, pipe in PIPELINES.items():
        if chunk_id in pipe.chunk_by_id:
            target_chunk = pipe.chunk_by_id[chunk_id]
            target_repo_id = r_id
            break

    if not target_chunk:
        raise HTTPException(status_code=404, detail="Chunk not found")

    hist = HISTORIES[target_repo_id]
    lineage = hist.get_symbol_lineage(target_chunk.name)

    # Compute diffs between consecutive steps in lineage
    history_steps = []
    prev_code = ""
    for step in lineage:
        code = step["code"]
        diff_text = "\n".join(difflib.unified_diff(prev_code.splitlines(), code.splitlines(), "before", "after", lineterm=""))
        history_steps.append({
            "sha": step["sha"],
            "date": step["date"],
            "message": step["message"],
            "path": step["path"],
            "start": step["start"],
            "end": step["end"],
            "code": code,
            "diff": diff_text if prev_code else "(initial version)",
        })
        prev_code = code

    return {
        "symbol": target_chunk.name,
        "versions_count": len(history_steps),
        "history": history_steps,
    }


@app.post("/webhooks/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """Handle incoming GitHub push webhook to trigger automatic incremental indexing."""
    payload = await request.json()
    repo_info = payload.get("repository", {})
    repo_name = repo_info.get("name", "").lower()
    head_sha = payload.get("after")

    matched_repo_id = None
    for r_id, r_data in REPOSITORIES.items():
        if repo_name and repo_name in r_id:
            matched_repo_id = r_id
            break

    if matched_repo_id and head_sha:
        hist = HISTORIES[matched_repo_id]
        pipe = PIPELINES[matched_repo_id]
        hist, pipe, stats = git_tracker.incremental_index(hist, pipe, REPOSITORIES[matched_repo_id]["local_path"], new_sha=head_sha)
        HISTORIES[matched_repo_id] = hist
        PIPELINES[matched_repo_id] = pipe
        return {"status": "processed", "repository": matched_repo_id, "stats": stats}

    return {"status": "ignored", "message": "No matching repository found or missing commit SHA"}


@app.post("/repositories/{repo_id}/sync")
def sync_repository(repo_id: str):
    """Check for new commits from Git and incrementally update if any changes exist."""
    repo = get_or_create_repo(repo_id)
    hist = HISTORIES[repo_id]
    pipe = PIPELINES[repo_id]

    hist, pipe, stats = git_tracker.incremental_index(hist, pipe, repo["local_path"])
    HISTORIES[repo_id] = hist
    PIPELINES[repo_id] = pipe
    repo["current_sha"] = hist.latest_commit["sha"]
    repo["last_sync"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {"status": "synced", "stats": stats}


@app.get("/evaluation")
def get_evaluation_results():
    """Return evaluation metrics or run evaluation on current sample."""
    csv_path = Path("ablation_results.csv")
    json_path = Path("evaluation_results.json")

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {"message": "Run /evaluation benchmark or execute python -m codetrace.evaluate"}
