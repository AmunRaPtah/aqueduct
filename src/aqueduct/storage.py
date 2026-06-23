"""Storage stage (bronze layer).

Loads every JSONL file in the landing zone into a raw DuckDB table, exactly as
ingested — no cleaning, no typing beyond what DuckDB infers. This is the durable
"source of truth" copy that downstream stages build on.
"""

from __future__ import annotations

import os
import time

import duckdb

from . import config

# Bound DuckDB's appetite. Its DEFAULT memory_limit is 80% of system RAM (~6.2 GB on
# this 7.8 GB box) with one thread per core — i.e. it assumes it owns the machine. On a
# shared box already running a media stack that overcommits RAM and the kernel OOM-kills
# the harvest mid-build (see the 2026-06-22 incident). Capping memory_limit makes DuckDB
# *spill its hash joins/aggregations to `temp_directory` on disk* instead of allocating
# past the ceiling, so the build stays within the harvest's 1.5 GB cgroup scope and never
# triggers a global OOM. Both knobs are env-overridable for one-off heavy manual runs.
_DB_MEMORY_LIMIT = os.environ.get("AQUEDUCT_DB_MEMORY_LIMIT", "1GB")
_DB_THREADS = os.environ.get("AQUEDUCT_DB_THREADS", "2")


def connect(retries: int = 8, wait: float = 4.0) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the warehouse, retrying on a transient write lock.

    The hourly harvest holds an exclusive lock while it builds; with incremental
    embedding that window is seconds, so a scheduled report/query just waits it out
    instead of failing.
    """
    config.ensure_dirs()
    # Spill to disk under the data dir (not CWD) when the memory cap is hit.
    cfg = {"memory_limit": _DB_MEMORY_LIMIT, "threads": _DB_THREADS,
           "temp_directory": str(config.DATA_DIR / ".duckdb_tmp")}
    if retries < 1:
        retries = 1
    last: Exception | None = None
    for _ in range(retries):
        try:
            return duckdb.connect(str(config.WAREHOUSE), config=cfg)
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
