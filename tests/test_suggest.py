"""Self-expanding watchlist: corpus insights -> new queries (offline, deterministic)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from aqueduct import harvest, suggest

NOW = datetime(2026, 6, 21, tzinfo=timezone.utc)

# A crafted facts object exercising every signal the generator mines.
FACTS = {
    "asymmetries": {
        "structurally_studied_but_undrugged": [
            ["HTR2A", 37, 0], ["TRPA1", 17, 1], ["DRD1", 24, 0],
        ],
    },
    "gaps": {
        "targets_without_drugs": [["KCNH2"], ["OPRK1"]],
        "drugs_without_papers": [["tapentadol"]],
    },
    "top_targets": [["HTR1A", 0, 28, 2]],
    "topics": {
        "mesh": [["Humans", 500], ["Receptors, Opioid", 40], ["Analgesics, Opioid", 33]],
        "concepts": [["Biology", 300], ["Persistent homology", 21]],
    },
}


def _patch_facts(monkeypatch, facts=FACTS):
    monkeypatch.setattr(suggest.analysis, "facts", lambda con=None: facts)


def _write_topics(path, **sources):
    path.write_text(json.dumps({
        "documents": {"europepmc": sources.get("europepmc", []),
                      "openalex": sources.get("openalex", [])},
        "structured": {"uniprot": sources.get("uniprot", []),
                       "pubchem": sources.get("pubchem", [])},
    }))


def test_generates_queries_from_insights(tmp_path, monkeypatch):
    _patch_facts(monkeypatch)
    tp = tmp_path / "topics.json"
    _write_topics(tp)

    added = suggest.generate(None, tp, per_source_cap=3, total_cap=40, now=NOW)
    by_source = {}
    for c in added:
        by_source.setdefault(c["source"], []).append(c["query"])

    # undrugged structural targets become both literature and uniprot queries
    assert any("HTR2A" in q for q in by_source.get("europepmc", []))
    assert "HTR2A" in by_source.get("uniprot", [])
    # generic MeSH/concepts are filtered; specific ones promoted
    assert all("Humans" not in q for q in by_source.get("europepmc", []))
    assert any("opioid" in q.lower() for q in by_source.get("europepmc", []))
    assert "Persistent homology" in by_source.get("openalex", [])
    assert "Biology" not in by_source.get("openalex", [])

    gen = json.loads(suggest.generated_path(tp).read_text())
    assert gen["_generated"] is True
    # every generated query carries provenance
    for c in added:
        assert f"{c['source']}\t{c['query']}" in gen["_provenance"]


def test_respects_per_source_cap(tmp_path, monkeypatch):
    _patch_facts(monkeypatch)
    tp = tmp_path / "topics.json"
    _write_topics(tp)
    added = suggest.generate(None, tp, per_source_cap=1, total_cap=40, now=NOW)
    counts = {}
    for c in added:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    assert all(n <= 1 for n in counts.values()), counts


def test_dedupes_against_curated_and_is_idempotent(tmp_path, monkeypatch):
    _patch_facts(monkeypatch)
    tp = tmp_path / "topics.json"
    # HTR2A already covered (curated) -> must not be re-proposed for those sources
    _write_topics(tp, uniprot=["HTR2A"], europepmc=["HTR2A receptor biology"])

    # high caps so the candidate pool is fully drained in one pass
    added1 = suggest.generate(None, tp, per_source_cap=99, total_cap=99, now=NOW)
    assert "HTR2A" not in [c["query"] for c in added1 if c["source"] == "uniprot"]
    assert all("htr2a" not in c["query"].lower() for c in added1 if c["source"] == "europepmc")

    # once candidates are exhausted, re-running over the same corpus is a no-op (converges)
    before = suggest.generated_path(tp).read_text()
    added2 = suggest.generate(None, tp, per_source_cap=99, total_cap=99, now=NOW)
    assert added2 == []
    assert suggest.generated_path(tp).read_text() == before


def test_total_cap(tmp_path, monkeypatch):
    _patch_facts(monkeypatch)
    tp = tmp_path / "topics.json"
    _write_topics(tp)
    added = suggest.generate(None, tp, per_source_cap=99, total_cap=2, now=NOW)
    assert len(added) == 2


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    _patch_facts(monkeypatch)
    tp = tmp_path / "topics.json"
    _write_topics(tp)
    added = suggest.generate(None, tp, dry_run=True, now=NOW)
    assert added
    assert not suggest.generated_path(tp).exists()


def test_load_topics_merges_generated(tmp_path, monkeypatch):
    _patch_facts(monkeypatch)
    tp = tmp_path / "topics.json"
    _write_topics(tp, europepmc=["opioid pharmacology"])
    suggest.generate(None, tp, now=NOW)

    merged = harvest.load_topics(tp)
    epmc = merged["documents"]["europepmc"]
    assert epmc[0] == "opioid pharmacology"             # curated first
    assert any("HTR2A" in q for q in epmc)              # generated merged in
    assert len(epmc) == len(set(q.lower() for q in epmc))  # no dupes
