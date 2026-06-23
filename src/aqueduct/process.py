"""Processing stage (silver + gold layers).

Turns raw bronze rows into clean, typed, deduplicated `events` (silver) and then
builds pre-aggregated `daily_metrics` (gold) ready for analytics. All transforms
run inside DuckDB — no data leaves the database.
"""

from __future__ import annotations

import duckdb

from .storage import connect


def process(con: duckdb.DuckDBPyConnection | None = None) -> dict[str, int]:
    """Build silver + gold tables from `events_raw`. Returns row counts."""
    owns = con is None
    con = con or connect()
    try:
        # --- silver: clean & type, drop dupes and malformed rows ---
        con.execute("DROP TABLE IF EXISTS events")
        con.execute(
            """
            CREATE TABLE events AS
            SELECT
                event_id,
                user_id,
                lower(event_type)        AS event_type,
                upper(country)           AS country,
                CAST(value AS DOUBLE)     AS value,
                CAST(ts AS TIMESTAMP)     AS ts,
                CAST(ts AS DATE)          AS event_date
            FROM (
                SELECT *, row_number() OVER (PARTITION BY event_id ORDER BY ts) AS rn
                FROM events_raw
                WHERE event_id IS NOT NULL AND ts IS NOT NULL
            )
            WHERE rn = 1
            """
        )

        # --- gold: daily metrics, the analytics-ready aggregate ---
        con.execute("DROP TABLE IF EXISTS daily_metrics")
        con.execute(
            """
            CREATE TABLE daily_metrics AS
            SELECT
                event_date,
                event_type,
                count(*)                          AS events,
                count(DISTINCT user_id)           AS unique_users,
                round(sum(value), 2)              AS revenue
            FROM events
            GROUP BY event_date, event_type
            ORDER BY event_date, event_type
            """
        )

        counts = {
            "events": con.execute("SELECT count(*) FROM events").fetchone()[0],
            "daily_metrics": con.execute(
                "SELECT count(*) FROM daily_metrics"
            ).fetchone()[0],
        }
        print(
            f"[process] built events (silver, {counts['events']} rows) "
            f"+ daily_metrics (gold, {counts['daily_metrics']} rows)"
        )
        return counts
    finally:
        if owns:
            con.close()
