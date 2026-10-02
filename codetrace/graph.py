"""Code Relationship Graph for CodeLens / CodeTrace AI.

Models relationships between code chunks:
- calls / called_by
- class membership (contains / member_of)
- imports
- data / call flow graph traversals
Supports fast structural retrieval and incremental index updates.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set
import networkx as nx

from .chunker import CodeChunk


class CodeGraph:
    """Lightweight directed code relationship graph."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.symbol_to_chunk_ids: Dict[str, Set[str]] = {}
        self.chunk_store: Dict[str, CodeChunk] = {}

    def build(self, chunks: List[CodeChunk]):
        """Build graph from a list of code chunks."""
        self.clear()
        self.add_chunks(chunks)

    def clear(self):
        self.graph.clear()
        self.symbol_to_chunk_ids.clear()
        self.chunk_store.clear()

    def add_chunks(self, chunks: List[CodeChunk]):
        """Incrementally add new or updated chunks to the graph."""
        # 1. Register nodes and symbol mapping
        for c in chunks:
            self.chunk_store[c.id] = c
            self.graph.add_node(
                c.id,
                name=c.name,
                path=c.path,
                start=c.start,
                end=c.end,
                parent_class=c.parent_class,
                commit=c.commit,
            )
            if c.name:
                self.symbol_to_chunk_ids.setdefault(c.name.lower(), set()).add(c.id)

        # 2. Build edges based on calls and class containment
        for c in chunks:
            # Class containment
            if c.parent_class:
                parent_sym = c.parent_class.lower()
                for pid in self.symbol_to_chunk_ids.get(parent_sym, []):
                    parent_chunk = self.chunk_store.get(pid)
                    if parent_chunk and parent_chunk.path == c.path:
                        self.graph.add_edge(pid, c.id, relation="CONTAINS")
                        self.graph.add_edge(c.id, pid, relation="MEMBER_OF")

            # Function calls
            for call_sym in c.calls:
                targets = self.symbol_to_chunk_ids.get(call_sym.lower(), set())
                for tid in targets:
                    if tid != c.id:
                        self.graph.add_edge(c.id, tid, relation="CALLS")
                        target_chunk = self.chunk_store.get(tid)
                        if target_chunk and c.name not in target_chunk.called_by:
                            target_chunk.called_by.append(c.name)

    def remove_chunks(self, chunk_ids: List[str]):
        """Incrementally remove chunks and their edges from the graph."""
        for cid in chunk_ids:
            if cid in self.chunk_store:
                c = self.chunk_store[cid]
                if c.name:
                    lower = c.name.lower()
                    if lower in self.symbol_to_chunk_ids:
                        self.symbol_to_chunk_ids[lower].discard(cid)
                        if not self.symbol_to_chunk_ids[lower]:
                            del self.symbol_to_chunk_ids[lower]
                del self.chunk_store[cid]
            if self.graph.has_node(cid):
                self.graph.remove_node(cid)

    def find_callers(self, symbol_name: str) -> List[CodeChunk]:
        """Return chunks that call the specified symbol."""
        targets = self.symbol_to_chunk_ids.get(symbol_name.lower(), set())
        callers = []
        for tid in targets:
            if self.graph.has_node(tid):
                for pred in self.graph.predecessors(tid):
                    edge_data = self.graph.get_edge_data(pred, tid, default={})
                    if edge_data.get("relation") == "CALLS":
                        callers.append(self.chunk_store[pred])
        return callers

    def find_callees(self, symbol_name: str) -> List[CodeChunk]:
        """Return chunks that are called by the specified symbol."""
        sources = self.symbol_to_chunk_ids.get(symbol_name.lower(), set())
        callees = []
        for sid in sources:
            if self.graph.has_node(sid):
                for succ in self.graph.successors(sid):
                    edge_data = self.graph.get_edge_data(sid, succ, default={})
                    if edge_data.get("relation") == "CALLS":
                        callees.append(self.chunk_store[succ])
        return callees

    def find_definitions(self, symbol_name: str) -> List[CodeChunk]:
        """Return chunks where symbol_name is defined."""
        cids = self.symbol_to_chunk_ids.get(symbol_name.lower(), set())
        return [self.chunk_store[cid] for cid in cids if cid in self.chunk_store]

    def get_neighborhood(self, chunk_id: str) -> Dict[str, List[str]]:
        """Return immediate callers, callees, parent class, and imports for Insight Panel."""
        c = self.chunk_store.get(chunk_id)
        if not c:
            return {"calls": [], "called_by": [], "parent_class": "", "imports": []}

        callers = list(dict.fromkeys(c.called_by))
        callees = list(dict.fromkeys(c.calls))

        # Check graph for any additional callers
        if self.graph.has_node(chunk_id):
            for pred in self.graph.predecessors(chunk_id):
                ed = self.graph.get_edge_data(pred, chunk_id, default={})
                if ed.get("relation") == "CALLS" and pred in self.chunk_store:
                    pred_name = self.chunk_store[pred].name
                    if pred_name and pred_name not in callers:
                        callers.append(pred_name)

        return {
            "calls": callees,
            "called_by": callers,
            "parent_class": c.parent_class or "",
            "imports": c.imports[:10],
        }

    def structural_score(
        self,
        chunk: CodeChunk,
        query_symbols: List[str],
        intent: str = "GENERAL_CODE_SEARCH",
    ) -> float:
        """Compute a structural retrieval score for chunk based on query symbols and intent."""
        if not query_symbols:
            return 0.1

        score = 0.0
        query_sym_lowers = [s.lower() for s in query_symbols]
        chunk_sym_lower = (chunk.name or "").lower()

        # 1. Direct symbol definition match
        if chunk_sym_lower in query_sym_lowers:
            score += 0.85

        # 2. Call-flow / Data-flow match
        # Check if chunk calls any query symbol
        calls_lower = [call.lower() for call in chunk.calls]
        overlap_calls = set(calls_lower) & set(query_sym_lowers)
        if overlap_calls:
            score += 0.50 if intent in ("CALL_FLOW", "DATA_FLOW", "DEPENDENCY") else 0.30

        # Check if chunk is called by any query symbol
        called_by_lower = [cb.lower() for cb in chunk.called_by]
        overlap_called = set(called_by_lower) & set(query_sym_lowers)
        if overlap_called:
            score += 0.55 if intent in ("CALL_FLOW", "DATA_FLOW", "DEPENDENCY") else 0.35

        # 3. Class membership match
        if chunk.parent_class and chunk.parent_class.lower() in query_sym_lowers:
            score += 0.40

        # 4. Partial substring or keyword match in symbols
        for qs in query_sym_lowers:
            if qs in chunk_sym_lower or chunk_sym_lower in qs:
                score += 0.25

        return min(1.0, score)
