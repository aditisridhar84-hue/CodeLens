"""Git repository ingestion, commit detection, diff analysis, and incremental indexing.

Supports:
- Cloning/fetching remote GitHub repositories
- Git diff analysis (Added, Modified, Deleted files)
- Webhook commit triggers & periodic polling commit detection
- Incremental index updates without rebuilding unchanged files
- Live simulated commit runner for demo workflows
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import git

from .chunker import CodeChunk, chunk_source_code
from .pipeline import Pipeline
from .versions import HistoryIndex, build_history, link_versions, save_sqlite


def normalize_git_sha(value: Optional[Any]) -> Optional[str]:
    """Coerce GitPython SHA values into a clean hex string.

    Git metadata can occasionally arrive as raw bytes or a decorated string.
    This helper prevents invalid bytes from being passed back into rev-parse.
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    value = str(value).strip()
    if not value:
        return None
    if value.startswith("b'") and value.endswith("'"):
        try:
            value = value[2:-1]
        except Exception:
            pass
    if value.startswith("b\"") and value.endswith("\""):
        value = value[2:-1]
    value = value.replace("\x00", "")
    if len(value) >= 7 and all(ch in "0123456789abcdefABCDEF" for ch in value[:40]):
        return value
    return None


@dataclass
class IndexStats:
    repository: str
    current_sha: str
    status: str
    files_count: int
    chunks_count: int
    functions_count: int
    embeddings_count: int
    last_sync: str
    index_time_secs: float
    last_commit_info: Dict[str, Any]
    diff_summary: Dict[str, Any]


class GitTracker:
    """Manages Git repository ingestion, commit monitoring, and incremental indexing."""

    def __init__(self, base_data_dir: str = "data/repos"):
        self.base_data_dir = Path(base_data_dir)
        self.base_data_dir.mkdir(parents=True, exist_ok=True)

    def is_github_url(self, path_or_url: str) -> bool:
        s = path_or_url.strip()
        return s.startswith("http://") or s.startswith("https://") or s.startswith("git@") or "github.com" in s

    def ingest_repository(
        self,
        repo_input: str,
        branch: Optional[str] = None,
        max_commits: int = 50,
    ) -> Tuple[str, str, git.Repo]:
        """Clone remote GitHub repo or open local repository.
        Returns: (local_path, current_sha, git.Repo instance)
        """
        repo_input = repo_input.strip()

        if self.is_github_url(repo_input):
            # Remote GitHub repository
            # Clean URL to determine folder name
            clean_name = repo_input.rstrip("/").split("/")[-1]
            if clean_name.endswith(".git"):
                clean_name = clean_name[:-4]
            dest_dir = self.base_data_dir / clean_name

            if dest_dir.exists() and (dest_dir / ".git").exists():
                # Already cloned, fetch updates
                repo = git.Repo(dest_dir)
                try:
                    repo.remotes.origin.fetch()
                    if branch:
                        repo.git.checkout(branch)
                        repo.remotes.origin.pull(branch)
                except Exception:
                    pass
            else:
                # Fresh clone
                clone_kwargs = {"depth": max_commits}
                if branch:
                    clone_kwargs["branch"] = branch
                repo = git.Repo.clone_from(repo_input, dest_dir, **clone_kwargs)

            local_path = str(dest_dir)
            current_sha = repo.head.commit.hexsha
            return local_path, current_sha, repo
        else:
            # Local directory
            resolved_path = str(Path(repo_input).resolve())
            try:
                repo = git.Repo(resolved_path, search_parent_directories=True)
                current_sha = repo.head.commit.hexsha
            except Exception:
                repo = None
                current_sha = "WORKTREE"
            return resolved_path, current_sha, repo

    def get_commit_diff(self, repo_path: str, old_sha: str, new_sha: str) -> Dict[str, List[str]]:
        """Compute git diff between two commits, categorizing files into added, modified, deleted."""
        summary = {"added": [], "modified": [], "deleted": []}
        old_sha = normalize_git_sha(old_sha)
        new_sha = normalize_git_sha(new_sha)
        if old_sha is None or new_sha is None or old_sha == new_sha or old_sha == "WORKTREE" or new_sha == "WORKTREE":
            return summary

        try:
            repo = git.Repo(repo_path, search_parent_directories=True)
            old_commit = repo.commit(old_sha)
            new_commit = repo.commit(new_sha)
            diff_index = old_commit.diff(new_commit)

            for diff in diff_index:
                if diff.new_file:
                    summary["added"].append(diff.b_path)
                elif diff.deleted_file:
                    summary["deleted"].append(diff.a_path)
                elif diff.renamed_file:
                    summary["deleted"].append(diff.a_path)
                    summary["added"].append(diff.b_path)
                else:
                    summary["modified"].append(diff.b_path)
        except Exception:
            pass

        return summary

    def poll_latest_sha(self, repo_path: str) -> Optional[str]:
        """Check the latest SHA of the repository."""
        try:
            repo = git.Repo(repo_path, search_parent_directories=True)
            sha = normalize_git_sha(repo.head.commit.hexsha)
            return sha
        except Exception:
            return None

    def incremental_index(
        self,
        hist: HistoryIndex,
        pipe: Pipeline,
        repo_path: str,
        new_sha: Optional[str] = None,
    ) -> Tuple[HistoryIndex, Pipeline, Dict[str, Any]]:
        """Incrementally update history and pipeline indexes for the latest commit.
        Only parses added/modified files and updates changed chunks.
        """
        t0 = time.perf_counter()
        used_head_fallback = False
        try:
            repo = git.Repo(repo_path, search_parent_directories=True)
            normalized_sha = normalize_git_sha(new_sha)
            if normalized_sha:
                try:
                    current_commit = repo.commit(normalized_sha)
                except Exception:
                    current_commit = repo.head.commit
                    used_head_fallback = True
            else:
                current_commit = repo.head.commit
        except Exception as exc:
            return hist, pipe, {"error": f"Could not resolve the requested commit or repository HEAD: {exc}", "updated": False}

        current_sha = normalize_git_sha(current_commit.hexsha) or "WORKTREE"
        old_sha = normalize_git_sha(hist.latest_commit["sha"]) or "WORKTREE"

        if old_sha == current_sha:
            return hist, pipe, {
                "updated": False,
                "message": f"Already up-to-date at commit {current_sha[:7]}",
                "sha": current_sha,
                "changed_files": 0,
            }

        diff = self.get_commit_diff(repo_path, old_sha, current_sha)
        affected_files = set(diff["added"] + diff["modified"] + diff["deleted"])

        # New commit index
        new_commit_idx = len(hist.commits)
        new_commit_meta = {
            "sha": current_sha,
            "date": current_commit.committed_datetime.strftime("%Y-%m-%d %H:%M"),
            "author": str(current_commit.author),
            "message": current_commit.message.strip().split("\n")[0],
        }
        hist.commits.append(new_commit_meta)

        # 1. Propagate presence for all chunks from previous commit whose files are NOT affected
        added_chunks_count = 0
        updated_chunks_count = 0
        deleted_chunks_count = 0
        new_chunks_to_index: List[CodeChunk] = []

        # Find active chunks in previous commit
        prev_commit_idx = new_commit_idx - 1
        for i, pres in enumerate(hist.presence):
            c = hist.chunks[i]
            if prev_commit_idx in pres:
                if c.path not in affected_files:
                    # Unchanged file: carry forward presence
                    pres.append(new_commit_idx)
                else:
                    if c.path in diff["deleted"]:
                        deleted_chunks_count += 1
                    else:
                        updated_chunks_count += 1

        # 2. Parse only added and modified files from the new commit
        files_to_parse = set(diff["added"] + diff["modified"])
        for blob in current_commit.tree.traverse():
            if blob.type == "blob" and blob.path in files_to_parse:
                src = blob.data_stream.read().decode("utf-8", errors="ignore")
                parsed = chunk_source_code(src, blob.path, commit=current_sha[:8], repo=hist.repo_name)
                for ch in parsed:
                    key = ch.id
                    if key in hist.index_of:
                        # Existing chunk version re-appeared
                        idx = hist.index_of[key]
                        if not hist.presence[idx] or hist.presence[idx][-1] != new_commit_idx:
                            hist.presence[idx].append(new_commit_idx)
                    else:
                        # Genuinely new chunk
                        idx = len(hist.chunks)
                        hist.chunks.append(ch)
                        hist.presence.append([new_commit_idx])
                        hist.index_of[key] = idx
                        new_chunks_to_index.append(ch)
                        added_chunks_count += 1

        # 3. Update the retrieval pipeline with only the new/changed chunks
        if new_chunks_to_index:
            pipe.update_chunks(new_chunks_to_index)

        # 4. Recompute evolutionary links
        links = link_versions(hist)
        save_sqlite(hist, links)

        elapsed = time.perf_counter() - t0
        stats = {
            "updated": True,
            "sha": current_sha,
            "used_head_fallback": used_head_fallback,
            "previous_sha": old_sha,
            "added_files": diff["added"],
            "modified_files": diff["modified"],
            "deleted_files": diff["deleted"],
            "added_chunks": added_chunks_count,
            "updated_chunks": updated_chunks_count,
            "deleted_chunks": deleted_chunks_count,
            "index_time_secs": elapsed,
            "status": "READY",
        }
        return hist, pipe, stats
