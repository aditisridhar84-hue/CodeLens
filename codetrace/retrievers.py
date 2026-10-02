"""Multi-signal retrievers for CodeLens / CodeTrace AI.

Includes:
1. FAISS CPU Dense Embedding Retriever with embedding caching
2. BM25 Lexical Retriever with code-aware tokenization
3. Structural Retriever interfacing with CodeGraph
4. Reciprocal Rank Fusion (RRF) & Min-Max score normalization
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional, Tuple
import numpy as np

from .chunker import CodeChunk, split_identifier

# Optional sentence-transformers & FAISS
SENTENCE_TRANSFORMERS_AVAILABLE = False
FAISS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SentenceTransformer = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None


def tokenize(text: str) -> List[str]:
    """Code-aware tokenizer splitting words, identifiers (snake_case and camelCase)."""
    if not text:
        return []
    words = re.findall(r"\b[A-Za-z0-9_]+\b", text)
    tokens = []
    for w in words:
        tokens.append(w.lower())
        sub = split_identifier(w)
        if len(sub) > 1:
            tokens.extend(sub)
    return [t for t in tokens if len(t) > 1]


class BM25Retriever:
    """Lexical code retriever using BM25Okapi."""

    def __init__(self, chunks: List[CodeChunk]):
        self.chunks = chunks
        self.corpus_tokens = [self._chunk_to_tokens(c) for c in chunks]
        if self.corpus_tokens and BM25Okapi:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    def _chunk_to_tokens(self, c: CodeChunk) -> List[str]:
        # Emphasize symbol name and path
        sym_toks = tokenize(c.name) * 3
        path_toks = tokenize(c.path) * 2
        calls_toks = [t for call in c.calls for t in tokenize(call)]
        kw_toks = c.keywords * 2
        body_toks = tokenize(c.code)
        doc_toks = tokenize(c.doc)
        return sym_toks + path_toks + kw_toks + calls_toks + doc_toks + body_toks

    def search(self, query: str, top_k: int = 50) -> List[Tuple[CodeChunk, float]]:
        """Retrieve top_k candidates with BM25 scores."""
        if not self.bm25 or not self.chunks:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        raw_scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(raw_scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(raw_scores[idx])
            if score > 0:
                results.append((self.chunks[idx], score))
        return results

    def update_index(self, chunks: List[CodeChunk]):
        """Rebuild or update the BM25 index."""
        self.chunks = chunks
        self.corpus_tokens = [self._chunk_to_tokens(c) for c in chunks]
        if self.corpus_tokens and BM25Okapi:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None


class DenseRetriever:
    """FAISS-based CPU semantic retriever with embedding caching."""

    def __init__(
        self,
        chunks: List[CodeChunk],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.model_name = model_name
        self.chunks = chunks
        self.embedding_cache: Dict[str, np.ndarray] = cache if cache is not None else {}
        self.model = None
        self.index = None
        self.dim = 384
        self._init_model()
        if self.chunks:
            self.build_index(self.chunks)

    def _init_model(self):
        if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer:
            try:
                self.model = SentenceTransformer(self.model_name, device="cpu")
                dim_func = getattr(self.model, "get_embedding_dimension", None) or getattr(self.model, "get_sentence_embedding_dimension", None)
                self.dim = dim_func() if dim_func else 384
            except Exception:
                self.model = None

    def _chunk_to_text(self, c: CodeChunk) -> str:
        doc = f" // {c.doc}" if c.doc else ""
        parent = f" class {c.parent_class}" if c.parent_class else ""
        calls = f" calls: {', '.join(c.calls[:6])}" if c.calls else ""
        return f"{c.path} {parent} {c.name}{doc}{calls}\n{c.code[:400]}"

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if self.model:
            embs = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                device="cpu",
            )
            return np.array(embs, dtype=np.float32)
        # Fallback pseudo-dense bag-of-words embedding if model unavailable
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, txt in enumerate(texts):
            for tok in tokenize(txt):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
                vecs[i, h] += 1.0
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm
        return vecs

    def build_index(self, chunks: List[CodeChunk]):
        """Encode chunks (using cache where possible) and construct FAISS CPU index."""
        self.chunks = chunks
        if not chunks:
            self.index = None
            return

        texts_to_encode = []
        indices_to_encode = []
        all_embeddings = np.zeros((len(chunks), self.dim), dtype=np.float32)

        for i, c in enumerate(chunks):
            c.embedding_id = i
            if c.code_hash in self.embedding_cache:
                all_embeddings[i] = self.embedding_cache[c.code_hash]
            else:
                texts_to_encode.append(self._chunk_to_text(c))
                indices_to_encode.append(i)

        if texts_to_encode:
            new_embs = self.encode_texts(texts_to_encode)
            for idx, emb in zip(indices_to_encode, new_embs):
                all_embeddings[idx] = emb
                self.embedding_cache[chunks[idx].code_hash] = emb

        if FAISS_AVAILABLE and faiss:
            self.index = faiss.IndexFlatIP(self.dim)
            faiss.normalize_L2(all_embeddings)
            self.index.add(all_embeddings)
        else:
            self.index = all_embeddings  # Array fallback

    def search(self, query: str, top_k: int = 50) -> List[Tuple[CodeChunk, float]]:
        """Dense semantic search returning top_k candidates and cosine similarities."""
        if not self.chunks:
            return []
        q_emb = self.encode_texts([query])[0]
        q_emb = q_emb.reshape(1, -1)

        if FAISS_AVAILABLE and self.index and isinstance(self.index, faiss.Index):
            faiss.normalize_L2(q_emb)
            scores, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.chunks):
                    results.append((self.chunks[idx], max(0.0, float(score))))
            return results
        elif isinstance(self.index, np.ndarray):
            sims = np.dot(self.index, q_emb.T).flatten()
            top_idx = np.argsort(sims)[::-1][:top_k]
            return [(self.chunks[i], max(0.0, float(sims[i]))) for i in top_idx]

        return []


def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Normalize score dictionary to [0, 1] range."""
    if not scores:
        return {}
    vals = list(scores.values())
    min_v, max_v = min(vals), max(vals)
    if max_v <= min_v or max_v <= 0:
        return {k: 1.0 / len(scores) for k in scores}
    return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[CodeChunk, float]]],
    k: int = 60,
) -> Dict[str, float]:
    """Compute standard RRF (Reciprocal Rank Fusion) across multiple candidate lists."""
    rrf_scores: Dict[str, float] = {}
    for r_list in ranked_lists:
        for rank, (chunk, _) in enumerate(r_list):
            rrf_scores[chunk.id] = rrf_scores.get(chunk.id, 0.0) + (1.0 / (k + rank + 1))
    return rrf_scores
