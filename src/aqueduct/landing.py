"""Landing-zone helpers: incremental, idempotent JSONL accumulation.

Connectors call `merge_jsonl` instead of overwriting, so repeated or varied fetches
*accumulate* in the landing zone (a later batch updates an existing record by key,
new records are added, prior batches are never lost).
"""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Callable, Iterable, Sequence


def _key(rec: dict, key: str | Sequence[str] | Callable[[dict], object]):
    if callable(key):
        return key(rec)
    if isinstance(key, (tuple, list)):
        return tuple(rec.get(k) for k in key)
    return rec.get(key)


def merge_jsonl(
    path: Path,
    records: Iterable[dict],
    key: str | Sequence[str] | Callable[[dict], object],
) -> tuple[int, int]:
    """Merge `records` into the JSONL at `path`, de-duplicating by `key`.

    Existing rows are loaded, new rows overwrite matching keys, and the file is
    rewritten in insertion order (existing first, then genuinely new). Returns
    (total_rows, added) where `added` counts keys not previously present.
    """
    merged: dict[object, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                merged[_key(r, key)] = r
    before = len(merged)
    for r in records:
        merged[_key(r, key)] = r
    with path.open("w") as f:
        for r in merged.values():
            f.write(json.dumps(r) + "\n")
    return len(merged), len(merged) - before
