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


def test_bad_id_format_still_reported_post_hoc(con, env):
    # A doc with a valid (non-empty) id that the ingest gate admits, but whose id
    # doesn't match the europepmc shape -> validate's post-hoc regex still flags it.
    seed.seed_document("BADID", title="A good title", sections=[("B", "text")])
    corpus.build(con)
    rep = validate.validate(con)
    assert not rep["ok"]
    assert rep["checks"]["bad_id_format"] == 1
    assert "BADID" in rep["examples"]["bad_id_format"]


def test_validate_no_corpus_is_graceful(con, env):
    rep = validate.validate(con)
    assert rep["ok"] and rep["n_documents"] == 0
