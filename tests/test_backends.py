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
