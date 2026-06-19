"""arXiv full-text injection + parsing (offline — no PDF download)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import seed

from aqueduct.sources import arxiv


def test_inject_fulltext_yields_body_section():
    entry = ET.fromstring(seed.ATOM_ENTRY)
    arxiv.inject_fulltext(entry, "Section one discusses the model. Results follow.")
    xml = ET.tostring(entry, encoding="unicode")

    out = arxiv.parse_atom(xml)
    types = [s["sec_type"] for s in out["sections"]]
    assert types == ["title", "abstract", "body"]
    body = next(s for s in out["sections"] if s["sec_type"] == "body")
    assert "model" in body["text"] and body["sec_title"] == "full text"


def test_no_fulltext_stays_abstract_only():
    out = arxiv.parse_atom(seed.ATOM_ENTRY)
    assert [s["sec_type"] for s in out["sections"]] == ["title", "abstract"]
