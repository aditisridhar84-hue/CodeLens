"""Root compatibility shim for versions.py forwarding to codetrace.versions."""

from codetrace.versions import (
    HistoryIndex,
    snapshot_history,
    build_history,
    link_versions,
    save_sqlite,
    load_sqlite,
)

__all__ = [
    "HistoryIndex",
    "snapshot_history",
    "build_history",
    "link_versions",
    "save_sqlite",
    "load_sqlite",
]
