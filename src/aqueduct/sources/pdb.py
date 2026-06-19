"""RCSB PDB connector (protein structures — structured data).

Enriches the PDB structures referenced by the proteins already fetched from UniProt:
reads their `pdb_ids` cross-references, then pulls title / method / resolution for each
from the RCSB Data API (keyless). `query` is accepted for CLI symmetry but ignored —
the structure set is driven by the UniProt landing file. `limit` caps how many.
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .. import config

ENTRY = "https://data.rcsb.org/rest/v1/core/entry"
USER_AGENT = "aqueduct/0.1 (data pipeline)"
DELAY = 0.1


def _get(url: str, *, retries: int = 3, timeout: int = 30) -> dict | None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception:  # noqa: BLE001 - 404 / transient: skip this id
            time.sleep(0.8 * (attempt + 1))
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
    n = 0
    with out.open("w") as f:
        for pid in ids:
            d = _get(f"{ENTRY}/{pid}")
            if not d:
                continue
            info = d.get("rcsb_entry_info", {})
            res = info.get("resolution_combined")
            f.write(
                json.dumps(
                    {
                        "pdb_id": pid,
                        "title": d.get("struct", {}).get("title"),
                        "method": info.get("experimental_method"),
                        "resolution": res[0] if isinstance(res, list) and res else None,
                        "fetched_at": fetched_at,
                    }
                )
                + "\n"
            )
            n += 1
            time.sleep(DELAY)
    print(f"[ingest]  pdb: {n} structures (from {len(ids)} UniProt refs) -> {out.relative_to(config.ROOT)}")
    return out
