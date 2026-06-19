"""Storage stage (bronze layer).

Loads every JSONL file in the landing zone into a raw DuckDB table, exactly as
ingested — no cleaning, no typing beyond what DuckDB infers. This is the durable
"source of truth" copy that downstream stages build on.
"""

from __future__ import annotations

import duckdb

from . import config


def connect() -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the warehouse database."""
    config.ensure_dirs()
    return duckdb.connect(str(config.WAREHOUSE))


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
