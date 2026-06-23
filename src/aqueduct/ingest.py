"""Ingestion stage.

Pulls events from a source and writes them to the landing zone as JSONL — one
file per ingestion batch. Here the "source" is a synthetic event generator so the
pipeline runs out of the box; swap `generate_events` for a real API/queue/file
reader and the rest of the pipeline is unchanged.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Iterator

from . import config

EVENT_TYPES = ["page_view", "click", "add_to_cart", "purchase", "signup"]
COUNTRIES = ["US", "GB", "DE", "FR", "BR", "IN", "JP", "CA", "AU", "NG"]


def generate_events(n: int, seed: int = 42) -> Iterator[dict]:
    """Yield `n` synthetic events spread over the last 7 days."""
    rng = random.Random(seed)
    now = datetime.now(timezone.utc)
    for i in range(n):
        ts = now - timedelta(seconds=rng.randint(0, 7 * 24 * 3600))
        etype = rng.choices(EVENT_TYPES, weights=[50, 25, 12, 5, 8])[0]
        yield {
            "event_id": f"evt_{i:08d}",
            "user_id": f"user_{rng.randint(1, max(1, n // 20)):06d}",
            "event_type": etype,
            "country": rng.choice(COUNTRIES),
            "value": round(rng.uniform(5, 250), 2) if etype == "purchase" else 0.0,
            "ts": ts.isoformat(),
        }


def ingest(n_events: int = 2000, seed: int = 42) -> Path:
    """Write a batch of events to the landing zone. Returns the file path."""
    config.ensure_dirs()
    # Deterministic batch name derived from the seed keeps re-runs idempotent.
    out = config.RAW_DIR / f"events_{seed:04d}.jsonl"
    with out.open("w") as f:
        for event in generate_events(n_events, seed=seed):
            f.write(json.dumps(event) + "\n")
    print(f"[ingest]  wrote {n_events} events -> {out.relative_to(config.ROOT)}")
    return out
