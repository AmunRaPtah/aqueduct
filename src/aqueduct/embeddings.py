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

    def fit(self, texts: list[str]) -> Embedder:
        return self

    def transform(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def state(self) -> dict:  # pragma: no cover - abstract
        raise NotImplementedError

    @classmethod
    def from_state(cls, state: dict) -> Embedder:  # pragma: no cover - abstract
        raise NotImplementedError


class LsaEmbedder(Embedder):
    """TF-IDF + truncated-SVD (latent semantic analysis) embedder."""

    name = "lsa"
    needs_fit = True

    # Above this many dense cells (n_docs × vocab) the TF-IDF matrix is built
    # sparsely. A dense float64 array of this size is ~32 MB; the real corpus is
    # ~100× larger (gigabytes) and ~99% zeros, so densifying it OOMs the box. Small
    # corpora (and all tests) stay on the exact dense path, byte-for-byte unchanged.
    _SPARSE_CELLS = 4_000_000

    def __init__(self, dims: int = 128, max_vocab: int = 8000, min_df: int = 2):
        self.dims = dims
        self.max_vocab = max_vocab
        self.min_df = min_df
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.components: np.ndarray | None = None  # (terms, k) term loadings V_k

    # --- fit / transform ---------------------------------------------------
    def fit(self, texts: list[str]) -> LsaEmbedder:
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

        tfidf = self._tfidf(doc_tokens)                   # (n_docs, terms), dense or sparse
        k = min(self.dims, min(tfidf.shape) - 1) if min(tfidf.shape) > 1 else 1
        vt = _truncated_vt(tfidf, k)                       # top-k right vectors (k, terms)
        self.components = vt[:k].T                          # (terms, k)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        tfidf = self._tfidf([_tokens(t) for t in texts])
        vecs = tfidf @ self.components                     # fold-in: x @ V_k
        return _l2norm(vecs)

    # --- internals ---------------------------------------------------------
    def _tfidf(self, doc_tokens: list[list[str]]):
        """Sparse for a large corpus, dense for a small one (same TF-IDF values)."""
        if len(doc_tokens) * max(len(self.vocab), 1) > self._SPARSE_CELLS:
            return self._tfidf_sparse(doc_tokens)
        return self._tfidf_matrix(doc_tokens)

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

    def _tfidf_sparse(self, doc_tokens: list[list[str]]) -> _CsrMatrix:
        """Same weighting as `_tfidf_matrix` (count → log1p → ×idf) in CSR form."""
        indptr = np.empty(len(doc_tokens) + 1, dtype=np.int64)
        indptr[0] = 0
        indices: list[int] = []
        data: list[float] = []
        for r, toks in enumerate(doc_tokens):
            counts: dict[int, float] = {}
            for w in toks:
                j = self.vocab.get(w)
                if j is not None:
                    counts[j] = counts.get(j, 0.0) + 1.0
            indices.extend(counts.keys())
            data.extend(counts.values())
            indptr[r + 1] = len(indices)
        cols = np.asarray(indices, dtype=np.int32)
        vals = np.asarray(data, dtype=np.float32)
        np.log1p(vals, out=vals)                           # sublinear tf
        vals *= self.idf[cols].astype(np.float32)          # idf weighting
        return _CsrMatrix((len(doc_tokens), len(self.vocab)), indptr, cols, vals)

    def state(self) -> dict:
        return {"dims": self.dims, "vocab": self.vocab,
                "idf": self.idf.tolist(), "components": self.components.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> LsaEmbedder:
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
        # bounded batch_size caps peak memory, so a full corpus re-embed (e.g. after a
        # chunking change) streams instead of allocating one giant activation tensor.
        v = self._model.encode(list(texts), normalize_embeddings=True,
                               batch_size=64, show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)

    def state(self) -> dict:
        return {"model": self.model_name}

    @classmethod
    def from_state(cls, state: dict) -> SentenceTransformerEmbedder:
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


class _CsrMatrix:
    """A minimal CSR sparse matrix — just enough for the LSA build.

    TF-IDF over a chunk corpus is ~99% zeros: a dense ``(n_chunks, vocab)`` array is
    gigabytes, while the sparse form is a few megabytes. Only the products the
    randomized SVD and the fold-in need are implemented (``m @ X`` and ``mᵀ @ X``);
    both stream in row blocks so the dense intermediate never exceeds one block.
    """

    __slots__ = ("shape", "dtype", "indptr", "indices", "data", "_block")

    def __init__(self, shape, indptr, indices, data, *, block: int = 256):
        self.shape = shape
        self.dtype = data.dtype
        self.indptr = indptr        # (n_rows + 1,) int64 row pointers
        self.indices = indices      # (nnz,) int32 column indices
        self.data = data            # (nnz,) float32 values
        self._block = block

    @property
    def T(self) -> _CsrT:
        return _CsrT(self)

    def __matmul__(self, X: np.ndarray) -> np.ndarray:
        """m @ X  — X is (n_cols, p), result (n_rows, p)."""
        X = np.asarray(X, dtype=np.float32)
        n_rows, p = self.shape[0], X.shape[1]
        out = np.empty((n_rows, p), dtype=np.float32)
        for s in range(0, n_rows, self._block):
            e = min(s + self._block, n_rows)
            a, b = self.indptr[s], self.indptr[e]
            contrib = self.data[a:b, None] * X[self.indices[a:b]]      # (block_nnz, p)
            local = np.repeat(np.arange(e - s), np.diff(self.indptr[s:e + 1]))
            blk = np.zeros((e - s, p), dtype=np.float32)
            np.add.at(blk, local, contrib)
            out[s:e] = blk
        return out

    def _tmatmul(self, Q: np.ndarray) -> np.ndarray:
        """mᵀ @ Q  — Q is (n_rows, p), result (n_cols, p)."""
        Q = np.asarray(Q, dtype=np.float32)
        p = Q.shape[1]
        out = np.zeros((self.shape[1], p), dtype=np.float32)
        for s in range(0, self.shape[0], self._block):
            e = min(s + self._block, self.shape[0])
            a, b = self.indptr[s], self.indptr[e]
            rows = np.repeat(np.arange(s, e), np.diff(self.indptr[s:e + 1]))
            np.add.at(out, self.indices[a:b], self.data[a:b, None] * Q[rows])
        return out


class _CsrT:
    """Lazy transpose view of a `_CsrMatrix` supporting only ``@`` (mᵀ @ X)."""

    __slots__ = ("_m",)

    def __init__(self, m: _CsrMatrix):
        self._m = m

    def __matmul__(self, Q: np.ndarray) -> np.ndarray:
        return self._m._tmatmul(Q)


def _randomized_vt(m, k: int, n_cols: int, *, oversample: int, n_iter: int) -> np.ndarray:
    """Randomized range-finder SVD; `m` may be a dense ndarray or a `_CsrMatrix`.

    Identical algorithm for both: only the matrix products differ in how they're
    evaluated (dense BLAS vs. streamed sparse), so the dense path stays byte-for-byte
    as before. ``Q.T @ m`` is computed as ``(mᵀ @ Q).T`` so the sparse type needs only
    a left-multiply.
    """
    rng = np.random.default_rng(0)
    p = k + oversample
    Q, _ = np.linalg.qr(m @ rng.standard_normal((n_cols, p)).astype(np.float32))  # (n_rows, p)
    for _ in range(n_iter):                                     # sharpen the subspace
        Q, _ = np.linalg.qr(m.T @ Q)                            # (n_cols, p)
        Q, _ = np.linalg.qr(m @ Q)                              # (n_rows, p)
    _, _, vt = np.linalg.svd((m.T @ Q).T, full_matrices=False)  # SVD of small (p, n_cols)
    return vt[:k].astype(np.float64)


def _truncated_vt(m, k: int, *, oversample: int = 10,
                  n_iter: int = 4) -> np.ndarray:
    """Top-k right singular vectors V^T (shape (k, n_features)).

    We only need the top k << min(shape) directions, so a *full* SVD is wasteful —
    on a large corpus it dominates the build and OOMs (it materialises all
    min(n_docs, n_terms) singular vectors). This uses a randomized range finder with
    power iterations (Halko, Martinsson & Tropp 2011): O(n·terms·k) work and a few
    small matrices instead of a full decomposition, in float32 (half the memory and
    BLAS time of float64, ample precision for LSA). For small matrices, where the
    approximation buys nothing and could be noisy, it falls back to an exact SVD — so
    the test corpora (and their results) are byte-for-byte unchanged. Seeded for
    reproducible builds.

    `m` is a dense ndarray for small corpora (and in tests); for a large corpus the
    caller passes a `_CsrMatrix` so the dense ``(n_docs, vocab)`` array — gigabytes,
    almost all zeros — is never materialised.
    """
    n_rows, n_cols = m.shape
    if isinstance(m, _CsrMatrix):                               # large, sparse corpus
        return _randomized_vt(m, k, n_cols, oversample=oversample, n_iter=n_iter)
    rank_cap = min(n_rows, n_cols)
    if rank_cap <= k + oversample or rank_cap <= 16:
        _, _, vt = np.linalg.svd(m, full_matrices=False)
        return vt[:k]
    m = np.asarray(m, dtype=np.float32)
    return _randomized_vt(m, k, n_cols, oversample=oversample, n_iter=n_iter)


# --- index persistence (sidecar under the data dir) ------------------------
def _index_paths():
    return (config.DATA_DIR / "lsa_model.json", config.DATA_DIR / "chunk_index.npz")


def _hash(text: str) -> str:
    import hashlib
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _corpus_hash(rows, hashes) -> str:
    """A stable fingerprint of the chunk corpus (ids + content hashes)."""
    return _hash("".join(f"{r[0]}:{r[1]}:{h}" for r, h in zip(rows, hashes, strict=True)))


def build_index(con=None, backend: str = "auto", dims: int = 128,
                model: str = "all-MiniLM-L6-v2", incremental: bool = True,
                force: bool = False) -> int:
    """Embed every chunk with the chosen backend and persist the index.

    backend='auto' picks 'st' when sentence-transformers is installed, else 'lsa'.
    With `incremental` and a pretrained backend (no global fit), vectors for unchanged
    chunks are reused from the prior index and only new/changed chunks are embedded —
    so a routine harvest re-embeds a handful of chunks, not the whole corpus. Fit-based
    backends (LSA) always rebuild fully, since their components are corpus-global.

    Idempotency: if the persisted index already covers this exact corpus (same backend
    + same `corpus_hash`), the build is skipped entirely unless `force=True`. So a
    harvest that lands nothing new costs one hash, not a full re-embed.
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
        hashes = [_hash(t) for t in texts]
        corpus_hash = _corpus_hash(rows, hashes)

        # auto-skip: the persisted index already matches this corpus + backend
        if not force:
            info = index_info()
            if (info.get("valid") and info.get("backend") == backend
                    and info.get("corpus_hash") == corpus_hash
                    and info.get("n_chunks") == len(rows)):
                print(f"[index]   corpus unchanged ({len(rows)} chunks) — skipping re-embed")
                return len(rows)

        opts = {"lsa": {"dims": dims}, "st": {"model": model}}.get(backend, {})
        emb = make_embedder(backend, **opts)
        model_path, idx_path = _index_paths()
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)

        # force => full rebuild (ignore reusable vectors); else reuse unchanged ones
        prev = (_load_reusable(idx_path, backend)
                if (incremental and not emb.needs_fit and not force) else None)
        if prev is None:
            if emb.needs_fit:
                emb.fit(texts)
            vecs = emb.transform(texts).astype(np.float32)
            reused = 0
        else:
            vecs, reused = _embed_incremental(emb, texts, hashes, rows, prev)

        model_path.write_text(json.dumps({
            "index_version": 1, "backend": emb.name, "state": emb.state(),
            "dims": int(vecs.shape[1]), "n_chunks": len(rows), "corpus_hash": corpus_hash,
        }))
        np.savez(idx_path, vectors=vecs, hashes=np.array(hashes),
                 pmcid=np.array([r[0] for r in rows]),
                 chunk_id=np.array([r[1] for r in rows]))
        note = f" ({reused} reused, {len(rows) - reused} new)" if reused else ""
        print(f"[index]   embedded {len(rows)} chunks via '{emb.name}' "
              f"({vecs.shape[1]} dims){note} -> {idx_path.name}")
        return len(rows)
    finally:
        if owns:
            con.close()


def _load_reusable(idx_path, backend: str):
    """Load a prior index if it matches the backend and carries per-chunk hashes."""
    model_path, _ = _index_paths()
    if not (idx_path.exists() and model_path.exists()):
        return None
    try:
        meta = json.loads(model_path.read_text())
        if meta.get("backend") != backend:
            return None
        data = np.load(idx_path, allow_pickle=True)
        if "hashes" not in data:
            return None
        return {(str(p), int(c)): (data["vectors"][i], str(data["hashes"][i]))
                for i, (p, c) in enumerate(zip(data["pmcid"], data["chunk_id"], strict=True))}
    except Exception:  # noqa: BLE001 - any issue -> full rebuild
        return None


def _embed_incremental(emb, texts, hashes, rows, prev) -> tuple[np.ndarray, int]:
    """Reuse vectors for unchanged (pmcid, chunk_id, hash); embed only the rest."""
    n = len(rows)
    todo_idx, todo_texts, reuse = [], [], {}
    for i, (r, h) in enumerate(zip(rows, hashes, strict=True)):
        hit = prev.get((str(r[0]), int(r[1])))
        if hit is not None and hit[1] == h:
            reuse[i] = hit[0]
        else:
            todo_idx.append(i)
            todo_texts.append(texts[i])
    dim = next(iter(reuse.values())).shape[0] if reuse else None
    new_vecs = emb.transform(todo_texts).astype(np.float32) if todo_texts else None
    if dim is None and new_vecs is not None:
        dim = new_vecs.shape[1]
    out = np.zeros((n, dim), dtype=np.float32)
    for i, v in reuse.items():
        out[i] = v
    for j, i in enumerate(todo_idx):
        out[i] = new_vecs[j]
    return out, len(reuse)


def index_info() -> dict:
    """Version/metadata of the persisted index (or {'exists': False})."""
    model_path, idx_path = _index_paths()
    if not (model_path.exists() and idx_path.exists()):
        return {"exists": False}
    try:
        meta = json.loads(model_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"exists": True, "valid": False}
    return {"exists": True, "valid": True,
            **{k: meta.get(k) for k in ("index_version", "backend", "dims",
                                        "n_chunks", "corpus_hash")}}


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
