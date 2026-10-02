"""Git history indexing + version linking + SQLite metadata store for CodeLens.

Preserves full version history across Git commits:
- Every unique chunk (file path + code) is indexed once.
- `presence[i]` records commit indices where chunk i is active.
- Evolutionary lineage tracks function transformations, renames, and diffs.
- SQLite persistence saves repository, commit, chunk, presence, and link data.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .chunker import CodeChunk, chunk_repo, chunk_source_code
from .retrievers import tokenize

SKIP_DIRS = (".venv/", "venv/", "node_modules/", "site-packages/", "__pycache__/", "dist/", "build/")
EXTENSIONS = (".py", ".js", ".ts", ".jsx", ".tsx", ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".java")


@dataclass
class HistoryIndex:
    chunks: List[CodeChunk]
    presence: List[List[int]]     # presence[i] -> list of commit indices where chunk i exists
    commits: List[Dict[str, Any]] # [{sha, date, author, message}]
    index_of: Dict[str, int] = field(default_factory=dict)
    repo_name: str = "local"
    repo_url: str = ""

    def __post_init__(self):
        self.index_of = {c.id: i for i, c in enumerate(self.chunks)}

    @property
    def latest_commit(self) -> Dict[str, Any]:
        return self.commits[-1] if self.commits else {"sha": "HEAD", "date": "", "author": "", "message": ""}

    def get_chunks_for_commit(self, commit_idx: int) -> List[CodeChunk]:
        """Return all chunks active as of a specific commit index."""
        active = []
        for i, pres in enumerate(self.presence):
            if commit_idx in pres:
                active.append(self.chunks[i])
        return active

    def get_symbol_lineage(self, symbol_name: str) -> List[Dict[str, Any]]:
        """Return chronological versions of a symbol across commits."""
        lineage = []
        sym_lower = symbol_name.lower()
        for i, c in enumerate(self.chunks):
            if c.name.lower() == sym_lower:
                pres = self.presence[i]
                for ci in pres:
                    cm = self.commits[ci]
                    lineage.append({
                        "commit_idx": ci,
                        "sha": cm["sha"],
                        "date": cm["date"],
                        "message": cm["message"],
                        "chunk": c,
                        "code": c.code,
                        "path": c.path,
                        "start": c.start,
                        "end": c.end,
                    })
        lineage.sort(key=lambda x: x["commit_idx"])
        return lineage


def snapshot_history(path: str, repo_name: str = "local") -> HistoryIndex:
    """Fallback for folders that are not git repos: one pseudo-commit."""
    chunks = chunk_repo(path, extensions=EXTENSIONS, commit="WORKTREE", repo_name=repo_name)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return HistoryIndex(
        chunks=chunks,
        presence=[[0] for _ in chunks],
        commits=[{"sha": "WORKTREE", "date": now, "author": "local", "message": "Working tree snapshot"}],
        repo_name=repo_name,
    )


def build_history(repo_path: str, max_commits: int = 100, repo_url: str = "") -> HistoryIndex:
    """Build full Git history index with presence tracking for every commit.

    Falls back to a single working-tree snapshot when the repo is empty, has no
    readable commits, or has stale metadata so the UI still indexes real files.
    """
    repo_name = Path(repo_path).resolve().name
    try:
        import git
        repo = git.Repo(repo_path, search_parent_directories=True)
    except Exception:
        return snapshot_history(repo_path, repo_name=repo_name)

    try:
        commits = list(repo.iter_commits(max_count=max_commits))[::-1]  # oldest -> newest
    except Exception:
        return snapshot_history(repo_path, repo_name=repo_name)

    if not commits:
        return snapshot_history(repo_path, repo_name=repo_name)

    blob_cache: Dict[str, List[CodeChunk]] = {}
    unique: Dict[str, CodeChunk] = {}
    presence: Dict[str, List[int]] = {}
    order: List[str] = []
    meta: List[Dict[str, Any]] = []

    for idx, c in enumerate(commits):
        c_date = c.committed_datetime.strftime("%Y-%m-%d %H:%M") if hasattr(c, "committed_datetime") else ""
        meta.append({
            "sha": c.hexsha,
            "date": c_date,
            "author": str(c.author),
            "message": c.message.strip().split("\n")[0] if c.message else "",
        })

        for blob in c.tree.traverse():
            if blob.type != "blob" or not any(blob.path.endswith(ext) for ext in EXTENSIONS):
                continue
            if any(s in blob.path + "/" for s in SKIP_DIRS):
                continue

            if blob.hexsha not in blob_cache:
                src = blob.data_stream.read().decode("utf-8", errors="ignore")
                blob_cache[blob.hexsha] = chunk_source_code(
                    src,
                    blob.path,
                    commit=c.hexsha[:8],
                    repo=repo_name,
                )

            for ch in blob_cache[blob.hexsha]:
                key = hashlib.sha1((ch.path + "\0" + ch.code).encode()).hexdigest()
                if key not in unique:
                    ch.id = key
                    ch.commit = c.hexsha[:8]
                    unique[key] = ch
                    presence[key] = []
                    order.append(key)
                if not presence[key] or presence[key][-1] != idx:
                    presence[key].append(idx)

    chunks_list = [unique[k] for k in order]
    presence_list = [presence[k] for k in order]

    history = HistoryIndex(
        chunks=chunks_list,
        presence=presence_list,
        commits=meta,
        repo_name=repo_name,
        repo_url=repo_url,
    )
    if not history.chunks:
        return snapshot_history(repo_path, repo_name=repo_name)
    return history


def _jaccard(a: List[str], b: List[str]) -> float:
    set_a, set_b = set(a), set(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def link_versions(hist: HistoryIndex, rename_threshold: float = 0.6) -> List[Dict[str, Any]]:
    """For each commit, link chunks that appeared to the chunks that vanished:
    same path+name -> 'modified'; otherwise best token-Jaccard >= threshold
    -> 'renamed/moved'. Chunks with no match are 'added'."""
    n = len(hist.commits)
    if n <= 1:
        return []

    by_commit: List[Set[int]] = [set() for _ in range(n)]
    for chunk_idx, pres in enumerate(hist.presence):
        for ci in pres:
            by_commit[ci].add(chunk_idx)

    toks: Dict[int, List[str]] = {}
    tk = lambda i: toks.setdefault(i, tokenize(hist.chunks[i].code))
    links: List[Dict[str, Any]] = []

    for i in range(1, n):
        appeared = by_commit[i] - by_commit[i - 1]
        vanished = set(by_commit[i - 1] - by_commit[i])

        for a in sorted(appeared):
            ca = hist.chunks[a]
            same = [v for v in vanished if hist.chunks[v].path == ca.path and hist.chunks[v].name == ca.name]
            if same:
                v = max(same, key=lambda v_idx: _jaccard(tk(a), tk(v_idx)))
                kind, sim = "modified", _jaccard(tk(a), tk(v))
            else:
                best = max(vanished, key=lambda v_idx: _jaccard(tk(a), tk(v_idx)), default=None)
                sim = _jaccard(tk(a), tk(best)) if best is not None else 0.0
                if best is not None and sim >= rename_threshold:
                    v, kind = best, "renamed/moved"
                else:
                    links.append({"commit_idx": i, "new": a, "old": None, "kind": "added", "sim": 0.0})
                    continue

            vanished.discard(v)
            links.append({"commit_idx": i, "new": a, "old": v, "kind": kind, "sim": round(sim, 3)})

        for v in vanished:
            links.append({"commit_idx": i, "new": None, "old": v, "kind": "removed", "sim": 0.0})

    return links


def save_sqlite(hist: HistoryIndex, links: List[Dict[str, Any]], db_path: str = "codetrace.db"):
    """Persist history, presence, commits, and evolution links to SQLite database."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS repositories (
            id TEXT PRIMARY KEY,
            url TEXT,
            latest_commit TEXT,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS commits (
            idx INTEGER,
            sha TEXT PRIMARY KEY,
            date TEXT,
            author TEXT,
            message TEXT
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            path TEXT,
            name TEXT,
            language TEXT,
            start INT,
            end INT,
            doc TEXT,
            code TEXT,
            calls TEXT,
            parent_class TEXT,
            first_commit TEXT
        );
        CREATE TABLE IF NOT EXISTS presence (
            chunk_id TEXT,
            commit_idx INT,
            PRIMARY KEY (chunk_id, commit_idx)
        );
        CREATE TABLE IF NOT EXISTS links (
            commit_idx INT,
            new_id TEXT,
            old_id TEXT,
            kind TEXT,
            sim REAL
        );
        CREATE TABLE IF NOT EXISTS index_status (
            repository_id TEXT PRIMARY KEY,
            commit_sha TEXT,
            status TEXT,
            files_count INT,
            chunks_count INT,
            index_time REAL,
            updated_at TEXT
        );
    """)

    # Clean existing data for this repo
    cur.execute("DELETE FROM commits")
    cur.execute("DELETE FROM chunks")
    cur.execute("DELETE FROM presence")
    cur.execute("DELETE FROM links")

    cur.executemany(
        "INSERT INTO commits VALUES(?,?,?,?,?)",
        [(i, m["sha"], m["date"], m["author"], m["message"]) for i, m in enumerate(hist.commits)],
    )

    cur.executemany(
        "INSERT OR REPLACE INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                c.id,
                c.path,
                c.name,
                c.language,
                c.start,
                c.end,
                c.doc,
                c.code,
                ",".join(c.calls),
                c.parent_class or "",
                c.commit,
            )
            for c in hist.chunks
        ],
    )

    cur.executemany(
        "INSERT OR REPLACE INTO presence VALUES(?,?)",
        [(hist.chunks[i].id, ci) for i, p in enumerate(hist.presence) for ci in p],
    )

    cid = lambda i: hist.chunks[i].id if (i is not None and i < len(hist.chunks)) else None
    cur.executemany(
        "INSERT INTO links VALUES(?,?,?,?,?)",
        [(l["commit_idx"], cid(l["new"]), cid(l["old"]), l["kind"], l["sim"]) for l in links],
    )

    con.commit()
    con.close()


def load_sqlite(db_path: str = "codetrace.db") -> Optional[Tuple[HistoryIndex, List[Dict[str, Any]]]]:
    """Load previously saved HistoryIndex and links from SQLite database."""
    if not os.path.exists(db_path):
        return None
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        commits_rows = cur.execute("SELECT idx, sha, date, author, message FROM commits ORDER BY idx ASC").fetchall()
        if not commits_rows:
            con.close()
            return None

        commits = [{"sha": r[1], "date": r[2], "author": r[3], "message": r[4]} for r in commits_rows]

        chunks_rows = cur.execute(
            "SELECT id, path, name, language, start, end, doc, code, calls, parent_class, first_commit FROM chunks"
        ).fetchall()
        chunks = []
        for r in chunks_rows:
            chunks.append(
                CodeChunk(
                    id=r[0],
                    path=r[1],
                    name=r[2],
                    language=r[3],
                    start=r[4],
                    end=r[5],
                    doc=r[6],
                    code=r[7],
                    calls=[c.strip() for c in r[8].split(",") if c.strip()],
                    parent_class=r[9] or None,
                    commit=r[10],
                )
            )

        chunk_id_to_idx = {c.id: i for i, c in enumerate(chunks)}
        presence_map: Dict[str, List[int]] = {c.id: [] for c in chunks}
        presence_rows = cur.execute("SELECT chunk_id, commit_idx FROM presence ORDER BY commit_idx ASC").fetchall()
        for cid, c_idx in presence_rows:
            if cid in presence_map:
                presence_map[cid].append(c_idx)

        presence = [presence_map[c.id] for c in chunks]

        links_rows = cur.execute("SELECT commit_idx, new_id, old_id, kind, sim FROM links").fetchall()
        links = []
        for r in links_rows:
            new_idx = chunk_id_to_idx.get(r[1]) if r[1] else None
            old_idx = chunk_id_to_idx.get(r[2]) if r[2] else None
            links.append({"commit_idx": r[0], "new": new_idx, "old": old_idx, "kind": r[3], "sim": r[4]})

        con.close()
        return HistoryIndex(chunks=chunks, presence=presence, commits=commits), links
    except Exception:
        return None
