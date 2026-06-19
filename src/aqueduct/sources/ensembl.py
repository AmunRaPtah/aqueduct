"""Ensembl connector (genomics — structured data).

Ensembl REST, keyless. Looks up gene records (location, biotype, description) by
symbol. With no `query`, it enriches the genes already present in the UniProt
landing file, so genomics joins the graph by gene symbol (drug -> target -> gene).
"""

from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from ..landing import merge_jsonl

BASE = "https://rest.ensembl.org"
USER_AGENT = "aqueduct/0.1 (data pipeline)"
SPECIES = "homo_sapiens"


def _get(url: str, *, retries: int = 3, timeout: int = 30) -> dict | None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception:  # noqa: BLE001 - unknown symbol / transient
            time.sleep(0.8 * (attempt + 1))
    return None


def _genes(query: str) -> list[str]:
    """Genes to look up: from the query (comma-separated) or the UniProt landing file."""
    if query and query.strip():
        return [g.strip() for g in query.split(",") if g.strip()]
    path = config.RAW_DIR / "uniprot" / "proteins.jsonl"
    if not path.exists():
        return []
    seen: dict[str, None] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            g = json.loads(line).get("gene")
            if g:
                seen.setdefault(g, None)
    return list(seen)


def _flatten(gene: str, d: dict) -> dict:
    return {
        "gene": gene,
        "ensembl_id": d.get("id"),
        "biotype": d.get("biotype"),
        "chromosome": d.get("seq_region_name"),
        "start": d.get("start"),
        "end": d.get("end"),
        "strand": d.get("strand"),
        "description": d.get("description"),
    }


def ingest(query: str = "", limit: int = 100) -> Path:
    """Land Ensembl gene records as JSONL (by query genes, or enriching UniProt)."""
    src_dir = config.raw_source_dir("ensembl")
    out = src_dir / "genes.jsonl"
    genes = _genes(query)[:limit]
    recs = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for g in genes:
        d = _get(f"{BASE}/lookup/symbol/{SPECIES}/{g}")
        if d and d.get("id"):
            recs.append({**_flatten(g, d), "fetched_at": fetched_at})
        time.sleep(0.12)
    total, added = merge_jsonl(out, recs, "gene")
    print(f"[ingest]  ensembl: +{added} new genes ({total} total, from {len(genes)} symbols) -> {out.relative_to(config.ROOT)}")
    return out
