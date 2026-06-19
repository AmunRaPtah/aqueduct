"""Semantic search over `doc_chunks`.

Pluggable embedder with a default **LSA** backend (TF-IDF -> truncated SVD, pure
NumPy): concept-level search with no API key and no model download. The index is a
derived sidecar artifact under the data dir; the warehouse stays the source of truth.

To swap in transformer / API embeddings later, implement the same two methods
(`fit`, `transform`) and point `build_index` at the new embedder — the storage and
search code is backend-agnostic.
"""

from __future__ import annotations

import json
import re

import numpy as np

from . import config
from .storage import connect

_TOKEN = re.compile(r"[a-z][a-z0-9]{2,}")  # words of >=3 chars
_STOP = {
    "the", "and", "for", "are", "was", "were", "with", "that", "this", "from",
    "have", "has", "had", "not", "but", "which", "their", "these", "those",
    "been", "also", "can", "may", "such", "than", "they", "our", "its", "into",
    "between", "using", "used", "use", "both", "each", "more", "most", "other",
    "results", "study", "studies", "showed", "shown", "however", "therefore",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class LsaEmbedder:
    """TF-IDF + truncated-SVD (latent semantic analysis) embedder."""

    def __init__(self, dims: int = 128, max_vocab: int = 8000, min_df: int = 2):
        self.dims = dims
        self.max_vocab = max_vocab
        self.min_df = min_df
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.components: np.ndarray | None = None  # (terms, k) term loadings V_k

    # --- fit / transform ---------------------------------------------------
    def fit(self, texts: list[str]) -> "LsaEmbedder":
        doc_tokens = [_tokens(t) for t in texts]
        df: dict[str, int] = {}
        for toks in doc_tokens:
            for w in set(toks):
                df[w] = df.get(w, 0) + 1
        # vocabulary: frequent-enough terms, capped by document frequency
        vocab = sorted((w for w, c in df.items() if c >= self.min_df),
                       key=lambda w: df[w], reverse=True)[: self.max_vocab]
        self.vocab = {w: i for i, w in enumerate(vocab)}
        n_docs = len(texts)
        self.idf = np.zeros(len(self.vocab), dtype=np.float64)
        for w, i in self.vocab.items():
            self.idf[i] = np.log((1 + n_docs) / (1 + df[w])) + 1.0

        tfidf = self._tfidf_matrix(doc_tokens)            # (n_docs, terms)
        k = min(self.dims, min(tfidf.shape) - 1) if min(tfidf.shape) > 1 else 1
        # truncated SVD via full SVD (corpora here are small)
        _, _, vt = np.linalg.svd(tfidf, full_matrices=False)
        self.components = vt[:k].T                          # (terms, k)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        tfidf = self._tfidf_matrix([_tokens(t) for t in texts])
        vecs = tfidf @ self.components                     # fold-in: x @ V_k
        return _l2norm(vecs)

    # --- internals ---------------------------------------------------------
    def _tfidf_matrix(self, doc_tokens: list[list[str]]) -> np.ndarray:
        m = np.zeros((len(doc_tokens), len(self.vocab)), dtype=np.float64)
        for r, toks in enumerate(doc_tokens):
            for w in toks:
                j = self.vocab.get(w)
                if j is not None:
                    m[r, j] += 1.0
        # sublinear tf then idf weighting
        np.log1p(m, out=m)
        m *= self.idf
        return m

    def state(self) -> dict:
        return {"dims": self.dims, "vocab": self.vocab,
                "idf": self.idf.tolist(), "components": self.components.tolist()}

    @classmethod
    def load(cls, state: dict) -> "LsaEmbedder":
        e = cls(dims=state["dims"])
        e.vocab = state["vocab"]
        e.idf = np.array(state["idf"])
        e.components = np.array(state["components"])
        return e


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-9)


# --- index persistence (sidecar under the data dir) ------------------------
def _index_paths():
    return (config.DATA_DIR / "lsa_model.json", config.DATA_DIR / "chunk_index.npz")


def build_index(con=None, dims: int = 128) -> int:
    """Embed every chunk and persist the index. Returns the chunk count."""
    owns = con is None
    con = con or connect()
    try:
        rows = con.execute(
            "SELECT pmcid, chunk_id, text FROM doc_chunks ORDER BY pmcid, chunk_id"
        ).fetchall()
        if not rows:
            print("[index]   no chunks to index — build the corpus first")
            return 0
        texts = [r[2] for r in rows]
        emb = LsaEmbedder(dims=dims).fit(texts)
        vecs = emb.transform(texts).astype(np.float32)
        model_path, idx_path = _index_paths()
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        model_path.write_text(json.dumps(emb.state()))
        np.savez(idx_path, vectors=vecs,
                 pmcid=np.array([r[0] for r in rows]),
                 chunk_id=np.array([r[1] for r in rows]))
        print(f"[index]   embedded {len(rows)} chunks ({vecs.shape[1]} dims) -> {idx_path.name}")
        return len(rows)
    finally:
        if owns:
            con.close()


def rank(query: str, k: int = 8) -> list[tuple[str, int, float]]:
    """Return the top-k (pmcid, chunk_id, score) for a query, or [] if no index."""
    model_path, idx_path = _index_paths()
    if not model_path.exists() or not idx_path.exists():
        return []
    emb = LsaEmbedder.load(json.loads(model_path.read_text()))
    data = np.load(idx_path, allow_pickle=True)
    vectors, pmcids, chunk_ids = data["vectors"], data["pmcid"], data["chunk_id"]
    qv = emb.transform([query])[0]
    sims = vectors @ qv
    top = np.argsort(sims)[::-1][:k]
    return [(str(pmcids[i]), int(chunk_ids[i]), float(sims[i])) for i in top]


def semantic_search(query: str, k: int = 8, con=None) -> None:
    """Print chunks ranked by cosine similarity to the query in LSA space."""
    owns = con is None
    con = con or connect()
    try:
        hits = rank(query, k=k)
        if not hits:
            print("No semantic index — run `corpus index` first.")
            return
        print(f"\n{k} semantic matches for {query!r}:\n")
        for pmcid, cid, score in hits:
            row = con.execute(
                "SELECT d.title, c.sec_title, c.text FROM doc_chunks c "
                "JOIN documents_raw d USING (pmcid) WHERE c.pmcid=? AND c.chunk_id=?",
                [pmcid, cid],
            ).fetchone()
            if not row:
                continue
            title, sec_title, text = row
            print(f"• [{score:.3f}] {pmcid} — {(title or '')[:55]}")
            print(f"    [{sec_title or 'body'}] {text[:150].strip()}…\n")
    finally:
        if owns:
            con.close()
