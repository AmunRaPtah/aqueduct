"""Semantic search / LSA embedder tests (offline, deterministic)."""

from __future__ import annotations

import numpy as np

from aqueduct import embeddings


def test_lsa_embedder_separates_topics():
    # two clearly distinct topics
    texts = [
        "opioid receptor agonist binds and activates analgesia signaling",
        "mu opioid receptor mediates pain relief via agonist binding",
        "matrix linear algebra eigenvalue decomposition vector space",
        "singular value decomposition factorizes a matrix into vectors",
    ]
    emb = embeddings.LsaEmbedder(dims=3, min_df=1).fit(texts)
    vecs = emb.transform(texts)
    # vectors are L2-normalized
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-6)
    # a query about pharmacology is closer to the opioid docs than the math docs
    q = emb.transform(["receptor agonist pain analgesia"])[0]
    sims = vecs @ q
    assert sims[0] > sims[2] and sims[1] > sims[3]


def test_embedder_state_roundtrip():
    emb = embeddings.LsaEmbedder(dims=2, min_df=1).fit(
        ["alpha beta gamma delta", "gamma delta epsilon zeta"])
    restored = embeddings.LsaEmbedder.load(emb.state())
    a = emb.transform(["alpha beta"])
    b = restored.transform(["alpha beta"])
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_build_index_and_rank(con, env):
    import seed
    from aqueduct import corpus
    seed.seed_document(
        "PMC_OP", title="Opioid analgesia",
        abstract="The opioid receptor agonist produces analgesia and pain relief.",
        sections=[("Body", "Receptor agonist binding triggers analgesic signaling. " * 8)],
    )
    seed.seed_document(
        "PMC_MATH", title="Matrix methods",
        abstract="Singular value decomposition factorizes a matrix into vectors.",
        sections=[("Body", "Eigenvalue decomposition of a matrix yields a vector basis. " * 8)],
    )
    corpus.build(con)
    n = embeddings.build_index(con, dims=8)
    assert n > 0
    hits = embeddings.rank("pain relief from receptor agonists", k=5)
    assert hits, "expected ranked hits"
    # the top hit should come from the opioid document, not the math one
    assert hits[0][0] == "PMC_OP"
