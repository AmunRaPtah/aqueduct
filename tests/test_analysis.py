"""Quantitative analysis library (offline, seeded graph)."""

from __future__ import annotations

import seed

from aqueduct import analysis, corpus, datasets, links


def _graph(con):
    seed.seed_document("PMC1", source="europepmc", doi="10.1/a",
                       title="Fentanyl and the mu receptor",
                       abstract="Fentanyl is a mu-opioid receptor agonist.",
                       sections=[("Body", "Duragesic delivers fentanyl. " * 6)])
    seed.seed_document("openalex:W1", source="openalex", doi="10.1/a",  # cross-source dup
                       title="Fentanyl and the mu receptor", keywords="Pharmacology; Chemistry")
    corpus.build(con)
    seed.seed_chembl(); seed.seed_clinicaltrials(); seed.seed_uniprot()
    seed.seed_pdb()
    datasets.build(con)
    links.build(con)


def test_overview_counts_unique_papers(con):
    _graph(con)
    o = analysis.overview(con)
    assert o["source_rows"] == 2 and o["unique_papers"] == 1   # deduped by DOI
    assert o["cross_source_dupes"] == 1


def test_top_drugs_and_targets(con):
    _graph(con)
    drugs = analysis.top_drugs(con)
    assert any(row[0] == "fentanyl" for row in drugs)
    targets = analysis.top_targets(con)
    assert any(row[0] == "OPRM1" for row in targets)


def test_gaps_structure(con):
    _graph(con)
    g = analysis.gaps(con)
    assert "targets_without_drugs" in g and "drugs_without_trials" in g


def test_facts_and_sheet(con):
    _graph(con)
    f = analysis.facts(con)
    assert set(f) >= {"overview", "trends", "top_drugs", "top_targets", "gaps"}
    sheet = analysis.facts_sheet(con)
    assert "Aqueduct facts" in sheet and "Top drugs" in sheet
    assert "fentanyl" in sheet.lower()
