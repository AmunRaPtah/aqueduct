"""Storage stage (bronze layer).

Loads every JSONL file in the landing zone into a raw DuckDB table, exactly as
ingested — no cleaning, no typing beyond what DuckDB infers. This is the durable
"source of truth" copy that downstream stages build on.
"""

from __future__ import annotations

import time

import duckdb

from . import config


def connect(retries: int = 8, wait: float = 4.0) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the warehouse, retrying on a transient write lock.

    The hourly harvest holds an exclusive lock while it builds; with incremental
    embedding that window is seconds, so a scheduled report/query just waits it out
    instead of failing.
    """
    config.ensure_dirs()
    last: Exception | None = None
    for _ in range(retries):
        try:
            return duckdb.connect(str(config.WAREHOUSE))
        except duckdb.IOException as e:  # lock held by another process
            last = e
            time.sleep(wait)
    raise last  # type: ignore[misc]


def store(con: duckdb.DuckDBPyConnection | None = None) -> int:
    """Load all landing-zone JSONL into `events_raw`. Returns the row count."""
    owns = con is None
    con = con or connect()
    try:
        pattern = str(config.RAW_DIR / "*.jsonl")
        con.execute("DROP TABLE IF EXISTS events_raw")
        con.execute(
            f"CREATE TABLE events_raw AS "
            f"SELECT * FROM read_json_auto('{pattern}', format='newline_delimited')"
        )
        rows = con.execute("SELECT count(*) FROM events_raw").fetchone()[0]
        print(f"[store]   loaded {rows} rows -> events_raw (bronze)")
        return rows
    finally:
        if owns:
            con.close()
