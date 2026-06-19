"""Pluggable embedding backends: registry, dispatch, persistence (offline)."""

from __future__ import annotations

import hashlib

import numpy as np

from aqueduct import embeddings


class _StubEmbedder(embeddings.Embedder):
    """A deterministic, pretrained-style backend (no fit, no heavy deps)."""

    name = "stub"
    needs_fit = False

    def __init__(self, dim: int = 16):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(
            int(hashlib.md5(text.encode()).hexdigest()[:8], 16))
        v = rng.standard_normal(self.dim)
        return v / (np.linalg.norm(v) or 1.0)

    def transform(self, texts):
        return np.vstack([self._vec(t) for t in texts]).astype(np.float32)

    def state(self):
        return {"dim": self.dim}

    @classmethod
    def from_state(cls, state):
        return cls(dim=state["dim"])


def test_default_backend_matches_availability():
    import importlib.util
    expected = "st" if importlib.util.find_spec("sentence_transformers") else "lsa"
    assert embeddings.default_backend() == expected


def test_st_model_cache_is_reused(monkeypatch):
    # populate the process cache; _ensure must reuse it without importing torch
    sentinel = object()
    monkeypatch.setitem(embeddings._ST_MODELS, "fake-model", sentinel)
    emb = embeddings.SentenceTransformerEmbedder(model="fake-model")
    emb._ensure()
    assert emb._model is sentinel


def test_auto_backend_resolves_in_build_index(con, env, monkeypatch):
    import json
    import seed
    from aqueduct import corpus
    # force 'lsa' so the test is deterministic and never loads a transformer
    monkeypatch.setattr(embeddings, "default_backend", lambda: "lsa")
    seed.seed_document("PMC1", abstract="alpha beta " * 4,
                       sections=[("B", "alpha beta gamma " * 6)])
    corpus.build(con)
    embeddings.build_index(con, backend="auto", dims=4)
    meta = json.loads((env / "lsa_model.json").read_text())
    assert meta["backend"] == "lsa"


def test_registry_dispatch_and_unknown():
    assert "lsa" in embeddings.BACKENDS and "st" in embeddings.BACKENDS
    assert isinstance(embeddings.make_embedder("lsa", dims=4), embeddings.LsaEmbedder)
    try:
        embeddings.make_embedder("nope")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_index_persists_backend_tag_and_rank_uses_it(con, env, monkeypatch):
    import seed
    from aqueduct import corpus
    monkeypatch.setitem(embeddings.BACKENDS, "stub", _StubEmbedder)

    seed.seed_document("PMC1", abstract="alpha beta gamma " * 4,
                       sections=[("B", "alpha beta gamma delta " * 8)])
    corpus.build(con)

    n = embeddings.build_index(con, backend="stub")
    assert n > 0
    # the index records which backend produced it
    import json
    meta = json.loads((env / "lsa_model.json").read_text())
    assert meta["backend"] == "stub"
    # rank reconstructs the stub backend from the tag and returns hits
    hits = embeddings.rank("alpha beta", k=3)
    assert hits and all(len(h) == 3 for h in hits)


def test_st_backend_lazy_import_message():
    """Without sentence-transformers installed, the 'st' backend errors clearly."""
    emb = embeddings.SentenceTransformerEmbedder()
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        try:
            emb.transform(["x"])
            assert False, "expected RuntimeError"
        except RuntimeError as e:
            assert "sentence-transformers" in str(e)
