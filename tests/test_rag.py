"""RAG retrieval surface (offline; LSA index for speed)."""

from __future__ import annotations

import seed

from aqueduct import corpus, datasets, embeddings, links, rag


def _setup(con):
    seed.seed_document("PMC1", title="Fentanyl analgesia",
                       abstract="Fentanyl is a potent mu-opioid receptor agonist.",
                       sections=[("Body", "Fentanyl produces analgesia via OPRM1. " * 6)])
    corpus.build(con)
    seed.seed_chembl(); seed.seed_uniprot()
    datasets.build(con)
    links.build(con)
    embeddings.build_index(con, backend="lsa", dims=8)


def test_retrieve_returns_grounded_chunks(con, env):
    _setup(con)
    out = rag.retrieve("opioid receptor agonist analgesia", k=5, con=con)
    assert out["n"] >= 1
    top = out["chunks"][0]
    assert top["id"] == "PMC1" and top["text"]
    assert set(top) >= {"id", "title", "doi", "source", "score", "text"}  # citeable


def test_retrieve_attaches_graph_context_for_known_drug(con, env):
    _setup(con)
    out = rag.retrieve("fentanyl mechanism of action", k=5, con=con)
    drugs = out.get("graph", {}).get("drugs", [])
    assert any(d["drug"] == "fentanyl" for d in drugs)
    assert "OPRM1" in drugs[0]["targets"]


def test_retrieve_empty_is_graceful(con, env):
    # no index built -> no crash, structured empty result
    seed.seed_document("PMC1")
    corpus.build(con)
    out = rag.retrieve("anything", con=con)
    assert out["n"] == 0 and "note" in out
