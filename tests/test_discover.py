"""Semantic-over-graph discovery tests (offline)."""

from __future__ import annotations

import seed

from aqueduct import corpus, datasets, discover, embeddings, links


def _setup(con):
    # a paper about the drug itself (lexical link), plus a paper about the TARGET's
    # biology that never names the drug (should surface semantically, not lexically)
    seed.seed_document(
        "PMC_DRUG", title="Fentanyl clinical use",
        abstract="Fentanyl is administered for analgesia.",
        sections=[("Body", "Clinicians titrate fentanyl carefully for pain. " * 6)],
    )
    seed.seed_document(
        "PMC_BIO", title="Mu receptor signaling",
        abstract="The mu opioid receptor couples to G proteins driving analgesia.",
        sections=[("Body",
                   "Mu opioid receptor agonist binding activates analgesic signaling cascades. " * 6)],
    )
    corpus.build(con)
    seed.seed_chembl()
    seed.seed_uniprot()
    datasets.build(con)
    links.build(con)
    embeddings.build_index(con, dims=8)


def test_profile_pulls_graph_context(con):
    _setup(con)
    prof = discover._profile(con, "fentanyl")
    assert "Mu-type opioid receptor" in prof["proteins"]
    assert any("agonist" in m.lower() for m in prof["mechanisms"])
    assert "fentanyl" in prof["text"].lower()


def test_discovery_surfaces_target_biology_paper(con):
    _setup(con)
    _, ranked = discover.ranked_papers("fentanyl", k=5, con=con)
    pmcids = [p for p, _s, _d in ranked]
    # the target-biology paper (never names fentanyl) is retrieved via the graph profile
    assert "PMC_BIO" in pmcids
    # and it is flagged SEMANTIC (not a direct lexical drug->doc link)
    assert any(p == "PMC_BIO" and not is_direct for p, _s, is_direct in ranked)
