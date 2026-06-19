"""Analytics stage.

Runs a handful of analytical queries over the modeled tables and prints a small
report. These are plain SQL — point a BI tool or notebook at the same warehouse
file to go further.
"""

from __future__ import annotations

import duckdb

from .storage import connect


def _fmt(rows: list[tuple], headers: tuple[str, ...]) -> str:
    widths = [
        max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
        for i, h in enumerate(headers)
    ]
    line = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = "\n".join(
        "  ".join(str(r[i]).ljust(widths[i]) for i in range(len(headers))) for r in rows
    )
    return f"{line}\n{sep}\n{body}"


def report(con: duckdb.DuckDBPyConnection | None = None) -> None:
    """Print the analytics summary."""
    owns = con is None
    con = con or connect()
    try:
        print("\n========== Aqueduct analytics ==========\n")

        totals = con.execute(
            """
            SELECT
                count(*)                AS total_events,
                count(DISTINCT user_id) AS unique_users,
                round(sum(value), 2)    AS total_revenue
            FROM events
            """
        ).fetchone()
        print(
            f"Events: {totals[0]:,}   Users: {totals[1]:,}   "
            f"Revenue: ${totals[2]:,.2f}\n"
        )

        print("-- Events by type --")
        rows = con.execute(
            """
            SELECT event_type, count(*) AS events, count(DISTINCT user_id) AS users
            FROM events GROUP BY event_type ORDER BY events DESC
            """
        ).fetchall()
        print(_fmt(rows, ("event_type", "events", "users")))

        print("\n-- Top 5 countries by revenue --")
        rows = con.execute(
            """
            SELECT country, round(sum(value), 2) AS revenue, count(*) AS events
            FROM events GROUP BY country ORDER BY revenue DESC LIMIT 5
            """
        ).fetchall()
        print(_fmt(rows, ("country", "revenue", "events")))

        print("\n-- Daily revenue (gold table) --")
        rows = con.execute(
            """
            SELECT event_date, round(sum(revenue), 2) AS revenue, sum(events) AS events
            FROM daily_metrics GROUP BY event_date ORDER BY event_date
            """
        ).fetchall()
        print(_fmt(rows, ("event_date", "revenue", "events")))
        print("\n========================================\n")
    finally:
        if owns:
            con.close()
