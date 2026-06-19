"""Semantic search over `doc_chunks`.

Pluggable embedder with a default **LSA** backend (TF-IDF -> truncated SVD, pure
NumPy): concept-level search with no API key and no model download. The index is a
derived sidecar artifact under the data dir; the warehouse stays the source of truth.

To swap in transformer / API embeddings later, implement the same two methods
(`fit`, `transform`) and point `build_index` at the new embedder — the storage and
search code is backend-agnostic.
"""

from __future__ import annotations

import importlib.util
import json
import re

import numpy as np

from . import config
from .storage import connect

# process-wide cache of loaded transformer models (keyed by model name), so
# build_index and rank in the same process share one load instead of two.
_ST_MODELS: dict[str, object] = {}


def default_backend() -> str:
    """'st' when sentence-transformers is importable, else the keyless 'lsa'."""
    return "st" if importlib.util.find_spec("sentence_transformers") else "lsa"

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


class Embedder:
    """Backend interface. Implement `fit`/`transform`/`state`/`from_state`.

    `needs_fit=True` backends learn from the corpus (LSA); pretrained backends
    (transformers) set it False and ignore `fit`.
    """

    name = "base"
    needs_fit = True

    def fit(self, texts: list[str]) -> "Embedder":
        return self

    def transform(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def state(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    @classmethod
    def from_state(cls, state: dict) -> "Embedder":  # pragma: no cover - abstract
        raise NotImplementedError


class LsaEmbedder(Embedder):
    """TF-IDF + truncated-SVD (latent semantic analysis) embedder."""

    name = "lsa"
    needs_fit = True

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
    def from_state(cls, state: dict) -> "LsaEmbedder":
        e = cls(dims=state["dims"])
        e.vocab = state["vocab"]
        e.idf = np.array(state["idf"])
        e.components = np.array(state["components"])
        return e

    load = from_state  # backwards-compatible alias


class SentenceTransformerEmbedder(Embedder):
    """Pretrained sentence-transformer embeddings (optional; `pip install -e '.[st]'`).

    No corpus fitting — the model is loaded by name and encodes text directly. Higher
    quality than LSA at the cost of a heavyweight dependency + one-time model download.
    """

    name = "st"
    needs_fit = False

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model_name = model
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        cached = _ST_MODELS.get(self.model_name)
        if cached is not None:
            self._model = cached
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "the 'st' backend needs sentence-transformers: pip install -e '.[st]'"
            ) from e
        self._model = SentenceTransformer(self.model_name)
        _ST_MODELS[self.model_name] = self._model  # cache for reuse this process

    def transform(self, texts: list[str]) -> np.ndarray:
        self._ensure()
        v = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)

    def state(self) -> dict:
        return {"model": self.model_name}

    @classmethod
    def from_state(cls, state: dict) -> "SentenceTransformerEmbedder":
        return cls(model=state.get("model", "all-MiniLM-L6-v2"))


# backend registry — add a class here to make it selectable via `--backend`
BACKENDS: dict[str, type[Embedder]] = {
    "lsa": LsaEmbedder,
    "st": SentenceTransformerEmbedder,
}


def make_embedder(backend: str, **opts) -> Embedder:
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choices: {sorted(BACKENDS)}")
    return BACKENDS[backend](**opts)


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-9)


# --- index persistence (sidecar under the data dir) ------------------------
def _index_paths():
    return (config.DATA_DIR / "lsa_model.json", config.DATA_DIR / "chunk_index.npz")


def build_index(con=None, backend: str = "auto", dims: int = 128,
                model: str = "all-MiniLM-L6-v2") -> int:
    """Embed every chunk with the chosen backend and persist the index.

    backend='auto' picks 'st' when sentence-transformers is installed, else 'lsa'.
    """
    if backend in (None, "auto"):
        backend = default_backend()
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
        opts = {"lsa": {"dims": dims}, "st": {"model": model}}.get(backend, {})
        emb = make_embedder(backend, **opts)
        if emb.needs_fit:
            emb.fit(texts)
        vecs = emb.transform(texts).astype(np.float32)
        model_path, idx_path = _index_paths()
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        model_path.write_text(json.dumps({"backend": emb.name, "state": emb.state()}))
        np.savez(idx_path, vectors=vecs,
                 pmcid=np.array([r[0] for r in rows]),
                 chunk_id=np.array([r[1] for r in rows]))
        print(f"[index]   embedded {len(rows)} chunks via '{emb.name}' "
              f"({vecs.shape[1]} dims) -> {idx_path.name}")
        return len(rows)
    finally:
        if owns:
            con.close()


def rank(query: str, k: int = 8) -> list[tuple[str, int, float]]:
    """Return the top-k (pmcid, chunk_id, score) for a query, or [] if no index."""
    model_path, idx_path = _index_paths()
    if not model_path.exists() or not idx_path.exists():
        return []
    meta = json.loads(model_path.read_text())
    emb = BACKENDS[meta["backend"]].from_state(meta["state"])
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
