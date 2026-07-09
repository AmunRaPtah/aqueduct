"""Ingest-time quality gate: garbage is quarantined, never stored.

Unit tests call `check_document` directly and see the *production* thresholds
(they don't take the `env` fixture, which relaxes length mins for the pipeline
fixtures). The integration test relies on the always-on empty-title/empty-body
checks, which fire regardless of the length thresholds.
"""

from __future__ import annotations

import seed

from aqueduct import corpus, quality


def _doc(**over):
    base = {"pmcid": "PMC1", "title": "A genuine article on receptors",
            "abstract": "We examine mu opioid receptor signaling in detail here today.",
            "has_body": True, "pub_year": 2024}
    base.update(over)
    return base


def test_accepts_clean_document():
    v = quality.check_document(_doc(), body_words=500)
    assert v.ok and v.reasons == []


def test_rejects_missing_title():
    v = quality.check_document(_doc(title="   "), body_words=500)
    assert not v.ok and "missing_title" in v.reasons


def test_rejects_title_too_short():
    v = quality.check_document(_doc(title="Hi"), body_words=500)
    assert not v.ok and "title_too_short" in v.reasons


def test_rejects_empty_id():
    v = quality.check_document(_doc(pmcid=""), body_words=500)
    assert not v.ok and "bad_id" in v.reasons


def test_rejects_broken_body():
    # has_body claimed but only 3 words parsed -> truncated/broken fetch
    v = quality.check_document(_doc(abstract=""), body_words=3)
    assert not v.ok and "body_missing" in v.reasons


def test_rejects_empty_body_regardless_of_threshold():
    # An empty body while has_body is claimed is garbage even with the length min off.
    v = quality.check_document(_doc(abstract=""), body_words=0)
    assert not v.ok and "body_missing" in v.reasons


def test_rejects_no_indexable_content():
    v = quality.check_document(
        _doc(has_body=False, abstract="too short"), body_words=0)
    assert not v.ok and "no_indexable_content" in v.reasons


def test_rejects_year_out_of_range():
    v = quality.check_document(_doc(has_body=False, pub_year=3200), body_words=0)
    assert not v.ok and "year_out_of_range" in v.reasons


def test_rejects_retraction_notice():
    v = quality.check_document(
        _doc(title="RETRACTED: A study that was later pulled"), body_words=500)
    assert not v.ok and "withdrawn_or_retracted" in v.reasons


def test_paper_about_retractions_is_not_rejected():
    # merely discussing retraction is fine — the notice pattern is anchored to a prefix
    v = quality.check_document(
        _doc(title="Trends in retraction of biomedical research papers"), body_words=500)
    assert v.ok


def test_rejects_garbage_text():
    v = quality.check_document(
        _doc(title="��� 12 @@@ ### $$$ %%% ^^^ &&&",
             abstract="!!! ??? ... /// \\\\\\ +++"), body_words=0)
    assert not v.ok and "garbage_text" in v.reasons


def test_gate_quarantines_in_pipeline(con, env):
    seed.seed_document("PMC1", title="A genuine article on opioids",
                       abstract="Real abstract about mu receptors and signaling.",
                       sections=[("Intro", "Body text about receptors and ligands.")])
    seed.seed_document("PMC2", title="   ",  # empty title -> quarantined
                       sections=[("Intro", "Some real body text here about things.")])
    seed.seed_document("PMC3", title="A genuine looking title about biology",
                       abstract="", sections=[("", "")])  # empty body -> quarantined
    corpus.build(con)
    stored = {r[0] for r in con.execute("SELECT pmcid FROM documents_raw").fetchall()}
    quarantined = {r[0]: r[1] for r in con.execute(
        "SELECT pmcid, reasons FROM documents_rejected").fetchall()}
    assert stored == {"PMC1"}
    assert set(quarantined) == {"PMC2", "PMC3"}
    assert "missing_title" in quarantined["PMC2"]
    assert "body_missing" in quarantined["PMC3"]


def test_clean_pipeline_quarantines_nothing(con, env):
    seed.seed_document("PMC1", title="A genuine article on opioids",
                       abstract="Real abstract about mu receptors and signaling.",
                       sections=[("Intro", "Body text about receptors and ligands.")])
    corpus.build(con)
    assert con.execute("SELECT count(*) FROM documents_rejected").fetchone()[0] == 0
