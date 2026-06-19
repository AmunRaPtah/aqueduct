"""ChEMBL connector (drug discovery / medicinal chemistry — structured data).

Lands compound records as flat JSONL in the structured landing zone. ChEMBL's
REST API is keyless. This is the seed of the *structured* ingestion mode: records,
not documents. Bioactivities / targets / assays are natural later increments.
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

BASE = "https://www.ebi.ac.uk/chembl/api/data"
USER_AGENT = "aqueduct/0.1 (data pipeline)"
PAGE_DELAY = 0.2


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


def _f(v):
    """Coerce ChEMBL's stringy numerics to float (None-safe)."""
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _i(v):
    f = _f(v)
    return int(f) if f is not None else None


def _flatten(m: dict) -> dict:
    """Pick a flat, analysis-friendly, properly-typed subset of a molecule record."""
    props = m.get("molecule_properties") or {}
    struct = m.get("molecule_structures") or {}
    return {
        "chembl_id": m.get("molecule_chembl_id"),
        "pref_name": m.get("pref_name"),
        "molecule_type": m.get("molecule_type"),
        "max_phase": _f(m.get("max_phase")),
        "first_approval": _i(m.get("first_approval")),
        "oral": m.get("oral"),
        "parenteral": m.get("parenteral"),
        "topical": m.get("topical"),
        "mw": _f(props.get("full_mwt")),
        "alogp": _f(props.get("alogp")),
        "psa": _f(props.get("psa")),
        "hba": _i(props.get("hba")),
        "hbd": _i(props.get("hbd")),
        "ro5_violations": _i(props.get("num_ro5_violations")),
        "qed_weighted": _f(props.get("qed_weighted")),
        "smiles": struct.get("canonical_smiles"),
        "inchi_key": struct.get("standard_inchi_key"),
    }


# Synonym types worth keeping for name-based entity matching (trade/approved names).
GOOD_SYN_TYPES = {"TRADE_NAME", "INN", "BAN", "USAN", "FDA", "USP", "JAN", "BNF"}


def _synonyms(m: dict) -> list[dict]:
    """Whitelisted (type, name) synonyms for one molecule."""
    cid = m.get("molecule_chembl_id")
    rows, seen = [], set()
    for s in m.get("molecule_synonyms", []) or []:
        name = (s.get("molecule_synonym") or "").strip()
        styp = s.get("syn_type")
        if name and styp in GOOD_SYN_TYPES and (styp, name.lower()) not in seen:
            seen.add((styp, name.lower()))
            rows.append({"chembl_id": cid, "syn_type": styp, "name": name})
    return rows


def search_molecules(query: str, limit: int = 100) -> list[dict]:
    """Free-text search ChEMBL molecules; returns raw molecule records."""
    out: list[dict] = []
    offset = 0
    q = urllib.parse.quote(query)
    while len(out) < limit:
        page = min(100, limit - len(out))
        url = f"{BASE}/molecule/search?q={q}&format=json&limit={page}&offset={offset}"
        data = _get(url)
        mols = data.get("molecules", [])
        if not mols:
            break
        out.extend(mols)
        offset += len(mols)
        if len(mols) < page:
            break
        time.sleep(PAGE_DELAY)
    return out[:limit]


def fetch_mechanisms(chembl_ids: list[str]) -> list[dict]:
    """Mechanism-of-action rows (molecule -> target) for the given molecule ids."""
    rows: list[dict] = []
    for i in range(0, len(chembl_ids), 20):  # batch via the __in filter
        batch = ",".join(chembl_ids[i : i + 20])
        url = f"{BASE}/mechanism?molecule_chembl_id__in={batch}&format=json&limit=200"
        for m in _get(url).get("mechanisms", []):
            rows.append(
                {
                    "molecule_chembl_id": m.get("molecule_chembl_id"),
                    "target_chembl_id": m.get("target_chembl_id"),
                    "action_type": m.get("action_type"),
                    "mechanism_of_action": m.get("mechanism_of_action"),
                }
            )
        time.sleep(PAGE_DELAY)
    return rows


def ingest(query: str, limit: int = 100) -> Path:
    """Land ChEMBL molecules + synonyms + mechanisms as JSONL in the landing zone."""
    src_dir = config.raw_source_dir("chembl")
    molecules = search_molecules(query, limit=limit)
    mol_path = src_dir / "molecules.jsonl"
    syn_path = src_dir / "synonyms.jsonl"
    mech_path = src_dir / "mechanisms.jsonl"
    fetched_at = datetime.now(timezone.utc).isoformat()
    mols, syns, ids = [], [], []
    for m in molecules:
        flat = _flatten(m)
        if flat["chembl_id"]:
            ids.append(flat["chembl_id"])
        mols.append({**flat, "query": query, "fetched_at": fetched_at})
        syns.extend(_synonyms(m))
    mechanisms = fetch_mechanisms(ids)
    _, n_mol = merge_jsonl(mol_path, mols, "chembl_id")
    merge_jsonl(syn_path, syns, ("chembl_id", "syn_type", "name"))
    merge_jsonl(mech_path, mechanisms, ("molecule_chembl_id", "target_chembl_id", "action_type"))
    print(
        f"[ingest]  chembl: +{n_mol} new molecules ({len(syns)} synonyms, "
        f"{len(mechanisms)} mechanisms) for {query!r} -> {mol_path.relative_to(config.ROOT)}"
    )
    return mol_path
