"""UniProt connector (proteins / drug targets — structured data).

Fetches reviewed (Swiss-Prot) protein entries via the UniProtKB REST API (keyless),
including the cross-references that bridge to the rest of the graph: PDB structure
ids and the ChEMBL *target* id (which ChEMBL mechanism-of-action rows point at).
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .. import config

API = "https://rest.uniprot.org/uniprotkb/search"
USER_AGENT = "aqueduct/0.1 (data pipeline)"
FIELDS = "accession,id,protein_name,gene_names,organism_name,length,cc_function,xref_pdb,xref_chembl"


def _get(url: str, *, retries: int = 3, timeout: int = 30) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url}") from last


def _flatten(r: dict) -> dict:
    desc = r.get("proteinDescription", {})
    rec = desc.get("recommendedName") or (desc.get("submissionNames") or [{}])[0]
    pname = (rec.get("fullName") or {}).get("value")
    genes = r.get("genes", [])
    gene = (genes[0].get("geneName", {}) or {}).get("value") if genes else None
    function = None
    for c in r.get("comments", []):
        if c.get("commentType") == "FUNCTION" and c.get("texts"):
            function = c["texts"][0].get("value")
            break
    xr = r.get("uniProtKBCrossReferences", [])
    pdb = [x["id"] for x in xr if x.get("database") == "PDB"]
    chembl = next((x["id"] for x in xr if x.get("database") == "ChEMBL"), None)
    return {
        "accession": r.get("primaryAccession"),
        "entry_name": r.get("uniProtkbId"),
        "protein_name": pname,
        "gene": gene,
        "organism": r.get("organism", {}).get("scientificName"),
        "length": r.get("sequence", {}).get("length"),
        "function": function,
        "pdb_ids": "; ".join(pdb) or None,
        "chembl_target": chembl,
    }


def search(query: str, limit: int = 100, reviewed: bool = True) -> list[dict]:
    """Search UniProtKB (reviewed by default); returns flattened protein records."""
    q = f"({query})" + (" AND reviewed:true" if reviewed else "")
    params = urllib.parse.urlencode(
        {"query": q, "format": "json", "size": min(limit, 500), "fields": FIELDS}
    )
    data = _get(f"{API}?{params}")
    return [_flatten(r) for r in data.get("results", [])][:limit]


def ingest(query: str, limit: int = 100) -> Path:
    """Land UniProt proteins as JSONL in the structured landing zone."""
    src_dir = config.raw_source_dir("uniprot")
    records = search(query, limit=limit)
    out = src_dir / "proteins.jsonl"
    fetched_at = datetime.now(timezone.utc).isoformat()
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps({**r, "query": query, "fetched_at": fetched_at}) + "\n")
    print(f"[ingest]  uniprot: {len(records)} proteins for {query!r} -> {out.relative_to(config.ROOT)}")
    return out
