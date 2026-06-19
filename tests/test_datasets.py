"""Structured-mode dataset loading + typing tests."""

from __future__ import annotations

import seed

from aqueduct import datasets


def test_build_loads_all_present_sources(con):
    seed.seed_chembl()
    seed.seed_clinicaltrials()
    seed.seed_uniprot()
    seed.seed_pdb()
    counts = datasets.build(con)
    assert counts["chembl"] >= 2
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    for t in ("chembl_molecules", "chembl_synonyms", "chembl_mechanisms",
              "clinical_trials", "uniprot_proteins", "pdb_structures"):
        assert t in tables


def test_chembl_mw_is_numeric(con):
    seed.seed_chembl()
    datasets.build(con)
    # would raise a BinderError if mw came in as VARCHAR
    avg = con.execute("SELECT round(avg(mw), 1) FROM chembl_molecules").fetchone()[0]
    assert avg > 0


def test_build_skips_absent_sources(con):
    seed.seed_chembl()  # only chembl present
    datasets.build(con)
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert "chembl_molecules" in tables
    assert "clinical_trials" not in tables
