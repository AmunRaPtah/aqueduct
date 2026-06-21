"""RCSB PDB connector (protein structures — structured data).

Enriches the PDB structures referenced by the proteins already fetched from UniProt:
reads their `pdb_ids` cross-references, then pulls title / method / resolution for each
from the RCSB Data API (keyless). `query` is accepted for CLI symmetry but ignored —
the structure set is driven by the UniProt landing file. `limit` caps how many.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .. import config, net
from ..landing import merge_jsonl

ENTRY = "https://data.rcsb.org/rest/v1/core/entry"
USER_AGENT = "aqueduct/0.1 (data pipeline)"
DELAY = 0.1


def _get(url: str, *, retries: int = 3, timeout: int = 30) -> dict | None:
    """Fetch JSON, returning None on any network failure (a 404 skips this id)."""
    try:
        return net.get_json(url, timeout=timeout, retries=retries)
    except net.NetworkError:
        return None


def _referenced_pdb_ids() -> list[str]:
    """Distinct PDB ids referenced by the UniProt proteins in the landing zone."""
    path = config.RAW_DIR / "uniprot" / "proteins.jsonl"
    if not path.exists():
        return []
    seen: dict[str, None] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        ids = (json.loads(line).get("pdb_ids") or "")
        for pid in ids.split(";"):
            pid = pid.strip()
            if pid:
                seen.setdefault(pid, None)
    return list(seen)


def ingest(query: str | None = None, limit: int = 100) -> Path:
    """Land structure metadata for UniProt-referenced PDB ids (capped at `limit`)."""
    src_dir = config.raw_source_dir("pdb")
    ids = _referenced_pdb_ids()[:limit]
    out = src_dir / "structures.jsonl"
    fetched_at = datetime.now(timezone.utc).isoformat()
    recs = []
    for pid in ids:
        d = _get(f"{ENTRY}/{pid}")
        if not d:
            continue
        info = d.get("rcsb_entry_info", {})
        res = info.get("resolution_combined")
        recs.append({
            "pdb_id": pid,
            "title": d.get("struct", {}).get("title"),
            "method": info.get("experimental_method"),
            "resolution": res[0] if isinstance(res, list) and res else None,
            "fetched_at": fetched_at,
        })
        time.sleep(DELAY)
    total, added = merge_jsonl(out, recs, "pdb_id")
    print(f"[ingest]  pdb: +{added} new structures ({total} total, from {len(ids)} UniProt refs) -> {out.relative_to(config.ROOT)}")
    return out
