"""PubChem connector (compounds / cheminformatics — structured data).

PUG REST, keyless. Resolves a name to CIDs, then batch-fetches properties. The
InChIKey bridges PubChem compounds to ChEMBL molecules in the link layer.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from ..landing import merge_jsonl

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
USER_AGENT = "aqueduct/0.1 (data pipeline)"
PROPS = "MolecularFormula,MolecularWeight,CanonicalSMILES,XLogP,InChIKey,IUPACName"


def _get(url: str, *, retries: int = 3, timeout: int = 30) -> dict | None:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception:  # noqa: BLE001 - not-found / transient
            time.sleep(0.8 * (attempt + 1))
    return None


def _f(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _flatten(p: dict, query: str) -> dict:
    return {
        "cid": p.get("CID"),
        "query": query,
        "iupac_name": p.get("IUPACName"),
        "molecular_formula": p.get("MolecularFormula"),
        "mw": _f(p.get("MolecularWeight")),
        "xlogp": _f(p.get("XLogP")),
        "smiles": p.get("CanonicalSMILES"),
        "inchi_key": p.get("InChIKey"),
    }


def ingest(query: str, limit: int = 100) -> Path:
    """Land PubChem compounds matching a name as JSONL."""
    src_dir = config.raw_source_dir("pubchem")
    out = src_dir / "compounds.jsonl"
    cids_doc = _get(f"{BASE}/compound/name/{urllib.parse.quote(query)}/cids/JSON")
    cids = (cids_doc or {}).get("IdentifierList", {}).get("CID", [])[:limit]
    recs = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for i in range(0, len(cids), 100):  # batch property lookups
        batch = ",".join(str(c) for c in cids[i : i + 100])
        doc = _get(f"{BASE}/compound/cid/{batch}/property/{PROPS}/JSON")
        for p in (doc or {}).get("PropertyTable", {}).get("Properties", []):
            recs.append({**_flatten(p, query), "fetched_at": fetched_at})
        time.sleep(0.25)
    total, added = merge_jsonl(out, recs, "cid")
    print(f"[ingest]  pubchem: +{added} new compounds ({total} total) for {query!r} -> {out.relative_to(config.ROOT)}")
    return out
