"""Orchestration: run the four stages in order over a single connection."""

from __future__ import annotations

from . import analytics, ingest, process, storage


def run(n_events: int = 2000, seed: int = 42) -> None:
    """Execute ingest -> store -> process -> analytics end-to-end."""
    print("=== Aqueduct pipeline ===")
    ingest.ingest(n_events=n_events, seed=seed)
    con = storage.connect()
    try:
        storage.store(con)
        process.process(con)
        analytics.report(con)
    finally:
        con.close()
    print("=== done ===")
