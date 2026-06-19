"""Incremental landing-zone accumulation (merge_jsonl)."""

from __future__ import annotations

import json

from aqueduct.landing import merge_jsonl


def _read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_merge_accumulates_and_dedupes(tmp_path):
    p = tmp_path / "d.jsonl"
    total, added = merge_jsonl(p, [{"id": "a", "v": 1}, {"id": "b", "v": 1}], "id")
    assert (total, added) == (2, 2)

    # second batch: one update (a) + one new (c); b is untouched and retained
    total, added = merge_jsonl(p, [{"id": "a", "v": 2}, {"id": "c", "v": 1}], "id")
    assert (total, added) == (3, 1)

    rows = {r["id"]: r["v"] for r in _read(p)}
    assert rows == {"a": 2, "b": 1, "c": 1}      # accumulated, a updated in place


def test_merge_composite_key(tmp_path):
    p = tmp_path / "syn.jsonl"
    merge_jsonl(p, [{"cid": "X", "name": "Foo"}, {"cid": "X", "name": "Bar"}], ("cid", "name"))
    total, added = merge_jsonl(p, [{"cid": "X", "name": "Foo"}], ("cid", "name"))
    assert total == 2 and added == 0            # exact duplicate, nothing added


def test_repeated_identical_fetch_is_idempotent(tmp_path):
    p = tmp_path / "d.jsonl"
    batch = [{"id": "a", "v": 1}, {"id": "b", "v": 2}]
    merge_jsonl(p, batch, "id")
    total, added = merge_jsonl(p, batch, "id")
    assert total == 2 and added == 0
    assert len(_read(p)) == 2
