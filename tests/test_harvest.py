"""BindingDB flatten + topic-driven harvest dispatch (offline)."""

from __future__ import annotations

from aqueduct import corpus, harvest
from aqueduct.sources import bindingdb


def test_bindingdb_affinities_and_flatten():
    resp = {"getLindsByUniprotsResponse": {"affinities": [
        {"query": "Mu receptor", "monomerid": "1", "smile": "CC",
         "affinity_type": "Ki", "affinity": "52", "doi": "10.x"},
    ]}}
    items = bindingdb.affinities(resp)
    assert len(items) == 1
    f = bindingdb._flatten("P35372", items[0])
    assert f["accession"] == "P35372" and f["affinity_nm"] == 52.0
    assert f["affinity_type"] == "Ki"
    # qualifier-prefixed values are coerced too
    assert bindingdb._f(">1000") == 1000.0
    assert bindingdb.affinities(None) == []


def test_harvest_dispatches_to_right_ingestors(monkeypatch):
    calls = []
    monkeypatch.setitem(corpus.INGESTORS, "openalex",
                        lambda q, limit=25: calls.append(("doc:openalex", q, limit)))
    monkeypatch.setitem(harvest.DATA_INGESTORS, "chembl",
                        lambda q, limit=25: calls.append(("data:chembl", q, limit)))

    topics = {"documents": {"openalex": ["q1", "q2"]},
              "structured": {"chembl": ["opioid"]}}
    result = harvest.harvest(topics, limit=7, build=False)

    assert result == {"documents": 2, "structured": 1}
    assert ("doc:openalex", "q1", 7) in calls
    assert ("doc:openalex", "q2", 7) in calls
    assert ("data:chembl", "opioid", 7) in calls


def test_harvest_skips_unknown_source(monkeypatch, capsys):
    harvest.harvest({"documents": {"nope": ["x"]}}, build=False)
    assert "unknown document source: nope" in capsys.readouterr().out
