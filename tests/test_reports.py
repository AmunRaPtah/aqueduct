"""Report composition (offline; LLM mocked so no API calls / tokens)."""

from __future__ import annotations

import seed

from aqueduct import corpus, reports


def _mini(con):
    seed.seed_document("PMC1", title="Fentanyl pharmacology",
                       abstract="Fentanyl acts on the mu-opioid receptor.")
    corpus.build(con)


def test_facts_only_when_llm_unavailable(con, env, monkeypatch):
    monkeypatch.setattr(reports.llm, "available", lambda: False)
    _mini(con)
    out = reports.generate(con=con)
    assert out["mode"] == "facts-only"
    assert "Data appendix" in out["markdown"]
    assert (env / "reports").exists() and out["path"].endswith(".md")


def test_single_call_report_uses_llm(con, env, monkeypatch):
    monkeypatch.setattr(reports.llm, "available", lambda: True)
    monkeypatch.setattr(reports.llm, "complete",
                        lambda *a, **k: "## Executive summary\nNARRATIVE-XYZ")
    _mini(con)
    out = reports.generate(con=con)
    assert out["mode"] == "single-call"
    assert "NARRATIVE-XYZ" in out["markdown"]
    assert "Data appendix" in out["markdown"]   # facts always appended


def test_report_prompt_includes_facts_and_topic():
    p = reports._prompt("FACTSHEET", "EXCERPT-1", "opioid receptors")
    assert "opioid receptors" in p and "FACTSHEET" in p and "EXCERPT-1" in p
