"""OpenAlex + patents parsers and pipeline integration (offline)."""

from __future__ import annotations

import json

from aqueduct import corpus, documents
from aqueduct.sources import openalex, patents


def test_reconstruct_abstract_from_inverted_index():
    inv = {"Mathematical": [0], "models": [1], "of": [2], "addiction": [3]}
    assert openalex.reconstruct_abstract(inv) == "Mathematical models of addiction"
    assert openalex.reconstruct_abstract(None) is None


def test_openalex_parse_sections():
    raw = json.dumps({"id": "W1", "title": "Topology of networks",
                      "abstract": "A study of graph topology."})
    out = openalex.parse(raw)
    assert [s["sec_type"] for s in out["sections"]] == ["title", "abstract"]


def test_patents_record_and_parse():
    p = {"patent_id": "123", "patent_title": "Battery electrode",
         "patent_abstract": "An improved electrode.", "patent_date": "2021-05-01",
         "assignees": [{"assignee_organization": "Acme"}],
         "cpc_current": [{"cpc_group_id": "H01M"}]}
    rec = patents._record(p)
    assert rec["pub_year"] == "2021" and rec["assignees"] == "Acme"
    assert "H01M" in rec["keywords"]
    out = patents.parse(json.dumps(rec))
    assert [s["sec_type"] for s in out["sections"]] == ["title", "abstract"]


def test_patents_no_key_is_graceful(env):
    # without PATENTSVIEW_API_KEY, ingest must no-op (not crash)
    patents.ingest("anything", limit=3)
    assert openalex.search  # smoke: module import path is wired


def test_openalex_flows_through_document_pipeline(con, env):
    import seed
    from aqueduct import config
    # simulate an ingested OpenAlex work landing in the corpus
    d = config.raw_source_dir("openalex")
    work = {"id": "W9", "title": "Emergent computation",
            "abstract": "We model emergence in complex systems and dynamics."}
    (d / "W9.json").write_text(json.dumps(work))
    rec = {"pmcid": "openalex:W9", "source": "openalex", "title": work["title"],
           "abstract": work["abstract"], "has_body": False,
           "xml_file": str(d / "W9.json"), "pub_year": 2024, "cited_by": 1}
    (d / "manifest.jsonl").write_text(json.dumps(rec) + "\n")

    corpus.build(con)
    n = con.execute(
        "SELECT count(*) FROM doc_sections WHERE pmcid='openalex:W9'").fetchone()[0]
    assert n >= 2  # title + abstract parsed via the openalex dispatch
