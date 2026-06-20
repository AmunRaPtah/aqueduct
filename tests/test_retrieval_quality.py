"""Retrieval-quality benchmark (audit #8) — a tiny gold set with Recall@k / MRR.

Seeds documents on distinct topics, builds the index, and asserts that known
queries retrieve their expected document near the top. Run before/after any change
to chunking or the embedding backend to catch regressions in retrieval precision.
"""

from __future__ import annotations

import seed

from aqueduct import corpus, embeddings, rag

# (query, expected pmcid that should rank in the top-k)
GOLD = [
    ("reversing opioid overdose with an antagonist", "PMC_NLX"),
    ("partial agonist maintenance therapy for dependence", "PMC_BUP"),
    ("eigenvalue decomposition of a matrix", "PMC_MATH"),
]


def _seed_corpus(con):
    seed.seed_document("PMC_NLX", title="Naloxone overdose reversal",
                       abstract="Naloxone is an opioid antagonist that reverses overdose.",
                       sections=[("Body", "The antagonist displaces agonists and restores breathing. " * 6)])
    seed.seed_document("PMC_BUP", title="Buprenorphine maintenance",
                       abstract="Buprenorphine, a partial agonist, maintains patients with opioid dependence.",
                       sections=[("Body", "Partial agonism reduces craving and withdrawal in maintenance. " * 6)])
    seed.seed_document("PMC_MATH", title="Matrix spectral methods",
                       abstract="Eigenvalue decomposition factorizes a matrix into eigenvectors.",
                       sections=[("Body", "Spectral decomposition of a matrix yields an eigenbasis. " * 6)])
    corpus.build(con)
    embeddings.build_index(con, backend="lsa", dims=16)


def test_recall_at_k_and_mrr(con, env):
    _seed_corpus(con)
    k = 3
    hits, recip = 0, 0.0
    for query, expected in GOLD:
        ranked = [c["id"] for c in rag.retrieve(query, k=k, con=con)["chunks"]]
        if expected in ranked:
            hits += 1
            recip += 1.0 / (ranked.index(expected) + 1)
    recall = hits / len(GOLD)
    mrr = recip / len(GOLD)
    # gold set is easy/separable — retrieval must be near-perfect or it regressed
    assert recall >= 0.66, f"Recall@{k}={recall:.2f} regressed"
    assert mrr >= 0.5, f"MRR={mrr:.2f} regressed"


def test_filters_narrow_results(con, env):
    _seed_corpus(con)
    # abstract-only filter returns only abstract chunks
    out = rag.retrieve("opioid antagonist", k=5, con=con, sec_types=["abstract"])
    assert all(c["sec_type"] == "abstract" for c in out["chunks"])
    # min_score filter drops everything when impossibly high
    assert rag.retrieve("opioid", k=5, con=con, min_score=2.0)["n"] == 0
