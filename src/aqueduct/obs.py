"""Structured logging + lightweight observability.

A tiny, dependency-free event log. Each `log(event, **fields)` call emits one
structured record; the sink is chosen by environment so it never disturbs the
human-readable `print` output the pipeline already produces:

    AQUEDUCT_LOG_JSON=1            -> one JSON object per line on stderr
    AQUEDUCT_LOG_FILE=/path/log    -> append JSON lines to a file (implies JSON)
    (neither set)                  -> silent (the default; tests stay quiet)

Records always carry an ISO-8601 `ts` and the `event` name. Use it for things
worth machine-reading later — retries, circuit trips, per-stage counts/timings —
without coupling to a logging framework.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone


def _enabled() -> bool:
    return bool(os.environ.get("AQUEDUCT_LOG_JSON") or os.environ.get("AQUEDUCT_LOG_FILE"))


def _sink():
    path = os.environ.get("AQUEDUCT_LOG_FILE")
    if path:
        return open(path, "a", encoding="utf-8")  # noqa: SIM115 - short-lived, flushed below
    return sys.stderr


def log(event: str, **fields) -> None:
    """Emit one structured record (no-op unless a JSON sink is configured)."""
    if not _enabled():
        return
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "event": event}
    # keep values JSON-serialisable; fall back to repr for the odd object
    for k, v in fields.items():
        try:
            json.dumps(v)
            rec[k] = v
        except (TypeError, ValueError):
            rec[k] = repr(v)
    line = json.dumps(rec, ensure_ascii=False)
    sink = _sink()
    try:
        sink.write(line + "\n")
        sink.flush()
    finally:
        if sink is not sys.stderr:
            sink.close()


@contextmanager
def span(event: str, **fields):
    """Time a block; emit `<event>` on entry and `<event>.done` with `ms` on exit.

    On exception, emits `<event>.error` with the exception type/message and re-raises.
    """
    start = time.monotonic()
    log(event, **fields)
    try:
        yield
    except Exception as e:  # noqa: BLE001 - observability only, always re-raised
        log(f"{event}.error", error_type=type(e).__name__, error=str(e),
            ms=round((time.monotonic() - start) * 1000, 1), **fields)
        raise
    else:
        log(f"{event}.done", ms=round((time.monotonic() - start) * 1000, 1), **fields)
