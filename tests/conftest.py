"""Shared fixtures: redirect Aqueduct's data paths to a temp dir, offline only."""

from __future__ import annotations

import pytest

from aqueduct import config, storage


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Point config at an isolated temp data dir; yields the data path."""
    data = tmp_path / "data"
    raw = data / "raw"
    raw.mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "WAREHOUSE", data / "warehouse.duckdb")
    # The pipeline fixtures use deliberately tiny synthetic docs; relax the ingest
    # gate's *length* thresholds so those pass. Empty-title/empty-body/garbage checks
    # stay active (they're what the gate exists for). quality.py reads these per-call.
    for name in ("MIN_TITLE_CHARS", "MIN_BODY_WORDS", "MIN_ABSTRACT_WORDS"):
        monkeypatch.setenv(f"AQUEDUCT_Q_{name}", "0")
    return data


@pytest.fixture
def con(env):
    """A DuckDB connection on the temp warehouse, closed at teardown."""
    c = storage.connect()
    yield c
    c.close()
