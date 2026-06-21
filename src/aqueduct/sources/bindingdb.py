"""BindingDB connector (binding affinities — structured data).

Quantitative drug–target binding (Ki / IC50 / Kd) per protein, via the keyless
BindingDB REST API. Complements ChEMBL mechanisms with measured potency. With no
`query`, enriches the UniProt accessions already in the landing zone.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from .. import config, net
from ..landing import merge_jsonl

API = "https://bindingdb.org/rest/getLigandsByUniprots"
USER_AGENT = "aqueduct/0.1 (data pipeline)"


def _get(url: str, *, retries: int = 3, timeout: int = 40) -> dict | None:
    """Fetch JSON, returning None on any network failure (a bad id skips, not fails)."""
    try:
        return net.get_json(url, timeout=timeout, retries=retries)
    except net.NetworkError:
        return None


def _f(v):
    try:
        return float(str(v).lstrip("<>=~ ")) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def affinities(data: dict | None) -> list[dict]:
    """Pull the affinities list out of BindingDB's (oddly-named) response object."""
    if not isinstance(data, dict):
        return []
    resp = next(iter(data.values()), {})
    return resp.get("affinities", []) if isinstance(resp, dict) else []


def _flatten(accession: str, a: dict) -> dict:
    return {
        "accession": accession,
        "target": a.get("query"),
        "monomer_id": a.get("monomerid"),
        "smiles": a.get("smile"),
        "affinity_type": a.get("affinity_type"),
        "affinity_nm": _f(a.get("affinity")),
        "pmid": a.get("pmid") or None,
        "doi": a.get("doi") or None,
    }


def _accessions(query: str) -> list[str]:
    if query and query.strip():
        return [a.strip() for a in query.split(",") if a.strip()]
    path = config.RAW_DIR / "uniprot" / "proteins.jsonl"
    if not path.exists():
        return []
    seen: dict[str, None] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            acc = json.loads(line).get("accession")
            if acc:
                seen.setdefault(acc, None)
    return list(seen)


def ingest(query: str = "", limit: int = 200) -> Path:
    """Land binding affinities as JSONL (by query accessions, or enriching UniProt)."""
    src_dir = config.raw_source_dir("bindingdb")
    out = src_dir / "affinities.jsonl"
    recs = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for acc in _accessions(query):
        if len(recs) >= limit:
            break
        params = urllib.parse.urlencode(
            {"uniprot": acc, "code": 0, "response": "application/json"})
        for a in affinities(_get(f"{API}?{params}")):
            recs.append({**_flatten(acc, a), "fetched_at": fetched_at})
            if len(recs) >= limit:
                break
        time.sleep(0.3)
    total, added = merge_jsonl(out, recs, ("accession", "monomer_id", "affinity_type"))
    print(f"[ingest]  bindingdb: +{added} new affinities ({total} total) -> {out.relative_to(config.ROOT)}")
    return out
