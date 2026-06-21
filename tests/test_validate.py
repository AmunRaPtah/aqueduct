"""Data-quality validation pass (offline)."""

from __future__ import annotations

import seed

from aqueduct import corpus, validate


def test_clean_corpus_has_no_issues(con, env):
    seed.seed_document("PMC1", title="Real title",
                       abstract="An abstract.", sections=[("Intro", "Some body text here.")])
    corpus.build(con)
    rep = validate.validate(con)
    assert rep["ok"] and rep["n_issues"] == 0 and rep["n_documents"] == 1


def test_flags_missing_title_and_bad_id(con, env):
    seed.seed_document("PMC1", title="Good", sections=[("B", "text")])
    # a malformed doc: blank title + an id that doesn't match the europepmc shape
    seed.seed_document("BADID", title="   ", sections=[("B", "text")])
    corpus.build(con)
    rep = validate.validate(con)
    assert not rep["ok"]
    assert rep["checks"]["missing_title"] == 1
    assert rep["checks"]["bad_id_format"] == 1
    assert "BADID" in rep["examples"]["bad_id_format"]


def test_flags_body_flag_without_sections(con, env):
    # has_body True but the body parses to nothing -> a broken full-text fetch
    seed.seed_document("PMC1", title="T", abstract="A", sections=[("", "")])
    corpus.build(con)
    assert con.execute(
        "SELECT count(*) FROM doc_sections WHERE pmcid='PMC1' AND sec_type='body'"
    ).fetchone()[0] == 0
    rep = validate.validate(con)
    assert rep["checks"]["body_flag_without_sections"] == 1


def test_validate_no_corpus_is_graceful(con, env):
    rep = validate.validate(con)
    assert rep["ok"] and rep["n_documents"] == 0
