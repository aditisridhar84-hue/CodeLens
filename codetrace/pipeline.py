"""Multi-Stage Code Retrieval Pipeline for CodeLens / CodeTrace AI.

Pipeline Stages:
1. Query Understanding & Intent Routing (intent, symbols, keywords, version)
2. Parallel Candidate Retrieval:
   - Dense Semantic Retrieval (FAISS CPU)
   - Lexical Retrieval (BM25)
   - Structural Retrieval (CodeGraph)
3. Multi-Signal Fusion (Normalized scores + tunable weights)
4. Code-Aware Reranker (Symbol match, path match, call flow, keyword overlap)
5. Confidence Second Pass (Full mode) -> Final Top-K
"""

from __future__ import annotations

import difflib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .chunker import CodeChunk
from .graph import CodeGraph
from .intent import QueryAnalysis, analyze_query
from .retrievers import BM25Retriever, DenseRetriever, min_max_normalize, reciprocal_rank_fusion

# Attempt to load custom weights from configs/weights.json if available
DEFAULT_WEIGHTS = {
    "semantic": 0.50,
    "lexical": 0.25,
    "structural": 0.25,
}


@dataclass
class SearchResult:
    chunk: CodeChunk
    score: float
    evidence: Dict[str, Any] = field(default_factory=dict)


class Pipeline:
    """End-to-end multi-signal retrieval pipeline."""

    def __init__(
        self,
        chunks: List[CodeChunk],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        weights: Optional[Dict[str, float]] = None,
    ):
        self.chunks = chunks
        self.chunk_by_id: Dict[str, CodeChunk] = {c.id: c for c in chunks}
        self.weights = weights or DEFAULT_WEIGHTS.copy()

        # Build index components
        self.dense_retriever = DenseRetriever(chunks, model_name=model_name)
        self.bm25_retriever = BM25Retriever(chunks)
        self.code_graph = CodeGraph()
        self.code_graph.build(chunks)

    def update_chunks(self, new_chunks: List[CodeChunk], removed_ids: Optional[List[str]] = None):
        """Incrementally update indexes for added/modified/removed code chunks."""
        removed_set = set(removed_ids or [])
        # Retain chunks not removed
        updated_chunks = [c for c in self.chunks if c.id not in removed_set]
        # Replace or append new chunks
        id_map = {c.id: idx for idx, c in enumerate(updated_chunks)}
        for nc in new_chunks:
            if nc.id in id_map:
                updated_chunks[id_map[nc.id]] = nc
            else:
                updated_chunks.append(nc)

        self.chunks = updated_chunks
        self.chunk_by_id = {c.id: c for c in self.chunks}

        # Update sub-indices
        self.dense_retriever.build_index(self.chunks)
        self.bm25_retriever.update_index(self.chunks)
        if removed_ids:
            self.code_graph.remove_chunks(removed_ids)
        self.code_graph.add_chunks(new_chunks)

    def search(
        self,
        query: str,
        mode: str = "full",
        k: int = 10,
        pool: int = 50,
        commit_filter: Optional[str] = None,
        language_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """Execute multi-stage code search according to selected pipeline mode.
        Modes:
        - 'dense': Mode A · Embedding only (baseline)
        - 'hybrid': Mode B · Hybrid BM25 + embeddings + RRF
        - 'rerank': Mode C · Hybrid + intent/structural rerank
        - 'full': Mode D · Full (+ confidence second pass)
        """
        if not self.chunks:
            return []

        # Stage A: Query Understanding
        q_analysis: QueryAnalysis = analyze_query(query)

        # Mode A: Dense baseline
        if mode == "dense":
            dense_cands = self.dense_retriever.search(query, top_k=pool)
            results = []
            for rank, (chunk, score) in enumerate(dense_cands):
                if commit_filter and chunk.commit != commit_filter:
                    continue
                if language_filter and chunk.language != language_filter:
                    continue
                results.append(
                    SearchResult(
                        chunk=chunk,
                        score=float(score),
                        evidence={"dense_score": round(score, 3), "mode": "dense"},
                    )
                )
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:k]

        # Stage B: Parallel Candidate Retrieval
        # 1. Semantic candidates
        dense_cands = self.dense_retriever.search(query, top_k=pool)
        dense_scores_raw = {c.id: score for c, score in dense_cands}

        # 2. Lexical candidates
        bm25_cands = self.bm25_retriever.search(query, top_k=pool)
        # Also query with extracted symbols to catch exact identifier matches
        if q_analysis.symbols:
            sym_query = " ".join(q_analysis.symbols)
            bm25_sym = self.bm25_retriever.search(sym_query, top_k=pool // 2)
            existing_cids = {cand.id for cand, _ in bm25_cands}
            for c, sc in bm25_sym:
                if c.id not in existing_cids:
                    bm25_cands.append((c, sc * 1.2))
                    existing_cids.add(c.id)
        bm25_scores_raw = {c.id: score for c, score in bm25_cands}

        # Union of candidate IDs
        candidate_ids = set(dense_scores_raw.keys()) | set(bm25_scores_raw.keys())
        # Also include definition chunks for any extracted query symbols
        for sym in q_analysis.symbols:
            for def_chunk in self.code_graph.find_definitions(sym):
                candidate_ids.add(def_chunk.id)

        if not candidate_ids:
            return []

        # Normalize dense and bm25 scores across candidate pool
        norm_dense = min_max_normalize(dense_scores_raw)
        norm_bm25 = min_max_normalize(bm25_scores_raw)

        # Mode B: Hybrid RRF
        if mode == "hybrid":
            rrf_scores = reciprocal_rank_fusion([dense_cands, bm25_cands], k=60)
            norm_rrf = min_max_normalize(rrf_scores)
            results = []
            for cid in candidate_ids:
                c = self.chunk_by_id.get(cid)
                if not c:
                    continue
                if commit_filter and c.commit != commit_filter:
                    continue
                if language_filter and c.language != language_filter:
                    continue
                score = norm_rrf.get(cid, 0.0)
                results.append(
                    SearchResult(
                        chunk=c,
                        score=score,
                        evidence={
                            "rrf_score": round(score, 3),
                            "dense_score": round(norm_dense.get(cid, 0.0), 3),
                            "bm25_score": round(norm_bm25.get(cid, 0.0), 3),
                            "mode": "hybrid",
                        },
                    )
                )
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:k]

        # Stage C: Structural Retrieval & Candidate Fusion
        # Compute structural score from CodeGraph for each candidate
        structural_scores: Dict[str, float] = {}
        for cid in candidate_ids:
            chunk = self.chunk_by_id[cid]
            structural_scores[cid] = self.code_graph.structural_score(
                chunk,
                query_symbols=q_analysis.symbols,
                intent=q_analysis.intent,
            )

        # Candidate fusion using configured weights
        w_dense = self.weights.get("semantic", 0.50)
        w_bm25 = self.weights.get("lexical", 0.25)
        w_struct = self.weights.get("structural", 0.25)

        candidates: List[SearchResult] = []
        for cid in candidate_ids:
            chunk = self.chunk_by_id[cid]
            if commit_filter and chunk.commit != commit_filter:
                continue
            if language_filter and chunk.language != language_filter:
                continue

            s_dense = norm_dense.get(cid, 0.0)
            s_bm25 = norm_bm25.get(cid, 0.0)
            s_struct = structural_scores.get(cid, 0.0)

            # Combined fusion score
            fusion_score = (w_dense * s_dense) + (w_bm25 * s_bm25) + (w_struct * s_struct)

            # Stage D: Code-Aware Reranking Signals
            sym_bonus = 0.0
            chunk_sym_lower = (chunk.name or "").lower()
            q_sym_lowers = [s.lower() for s in q_analysis.symbols]

            # Exact symbol name match bonus
            if chunk_sym_lower in q_sym_lowers:
                sym_bonus += 0.20
            elif any(qs in chunk_sym_lower for qs in q_sym_lowers):
                sym_bonus += 0.10

            # File name match bonus
            file_bonus = 0.0
            path_lower = chunk.path.lower()
            for kw in q_analysis.keywords:
                if len(kw) > 3 and kw in path_lower:
                    file_bonus += 0.05
                    break

            # Keyword overlap in code
            code_lower = chunk.code.lower()
            kw_matches = sum(1 for kw in q_analysis.keywords if kw in code_lower)
            kw_overlap_ratio = kw_matches / max(1, len(q_analysis.keywords))
            kw_bonus = min(0.15, kw_overlap_ratio * 0.15)

            # Call relationship bonus for CALL_FLOW or DATA_FLOW
            flow_bonus = 0.0
            if q_analysis.relationship_type in ("before", "calls"):
                # If query mentions a target (e.g. main) and this chunk is called by target
                if any(qs in [cb.lower() for cb in chunk.called_by] for qs in q_sym_lowers):
                    flow_bonus += 0.25
            elif q_analysis.relationship_type in ("after", "called_by"):
                if any(qs in [c_call.lower() for c_call in chunk.calls] for qs in q_sym_lowers):
                    flow_bonus += 0.25

            final_score = fusion_score + sym_bonus + file_bonus + kw_bonus + flow_bonus

            evidence = {
                "semantic": round(s_dense, 3),
                "lexical": round(s_bm25, 3),
                "structural": round(s_struct, 3),
                "symbol_match": round(sym_bonus, 3),
                "flow_bonus": round(flow_bonus, 3),
                "intent": q_analysis.intent,
                "second_pass": False,
            }
            candidates.append(SearchResult(chunk=chunk, score=final_score, evidence=evidence))

        candidates.sort(key=lambda r: r.score, reverse=True)

        # Mode C: Return hybrid + reranked candidates
        if mode == "rerank":
            return candidates[:k]

        # Mode D: Full Pipeline with Confidence Second Pass
        second_pass_triggered = False
        if len(candidates) >= 2:
            top_gap = candidates[0].score - candidates[1].score
            # If top result confidence is narrow (<0.08) or top score is low (<0.45)
            if top_gap < 0.08 or candidates[0].score < 0.45:
                second_pass_triggered = True
                # Second pass: expand query with callers, callees, and parent class of top candidates
                expanded_terms = set(q_analysis.keywords)
                for top_c in candidates[:3]:
                    expanded_terms.update(top_c.chunk.keywords[:3])
                    expanded_terms.update(top_c.chunk.calls[:2])

                expanded_query = query + " " + " ".join(list(expanded_terms)[:5])
                pass2_dense = self.dense_retriever.search(expanded_query, top_k=20)
                pass2_bm25 = self.bm25_retriever.search(expanded_query, top_k=20)

                pass2_scores = {c.id: sc for c, sc in pass2_dense}
                for c, sc in pass2_bm25:
                    pass2_scores[c.id] = pass2_scores.get(c.id, 0.0) + (sc * 0.1)

                for r in candidates:
                    if r.chunk.id in pass2_scores:
                        r.score += pass2_scores[r.chunk.id] * 0.15
                    r.evidence["second_pass"] = True

                candidates.sort(key=lambda r: r.score, reverse=True)

        if candidates:
            candidates[0].evidence["second_pass"] = second_pass_triggered

        return candidates[:k]
