"""SureChEMBL patents connector — keyless, via EMBL-EBI bulk Parquet over HTTP.

SureChEMBL (EMBL-EBI) chemically annotates ~160M patents. Its bulk data is CC BY 4.0
Parquet with no key and no login. Rather than download ~14 GB, we query the remote
Parquet directly with DuckDB's httpfs (HTTP range requests), driven by the InChIKeys of
the drugs/compounds already in our warehouse:

    compounds.parquet (inchi_key -> id)
      -> patent_compound_map.parquet (compound_id -> patent_id)
        -> patents.parquet (id -> number, title, date, assignee, cpc)

So for the compounds we already track, we land the patents that disclose them -- a
drug->patent edge for the graph -- with no API key. One batched 3-pass join per run, so
cost is roughly fixed (~7-8 min over the wire) regardless of how many compounds.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from .. import config
from ..landing import merge_jsonl

BASE = os.environ.get(
    "SURECHEMBL_PARQUET_BASE",
    "https://ftp.ebi.ac.uk/pub/databases/chembl/SureChEMBL/bulk_data/latest",
)
# The remote files are unsorted on inchi_key, so each pass scans a column over HTTP;
# cap how many patents we land per run so corpus growth (and runtime) stays bounded.
MAX_PATENTS = int(os.environ.get("SURECHEMBL_MAX_PATENTS", "500"))
_INCHIKEY = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def _warehouse_inchikeys() -> list[str]:
    """InChIKeys of the compounds we already track (ChEMBL + PubChem)."""
    if not config.WAREHOUSE.exists():
        return []
    con = duckdb.connect(str(config.WAREHOUSE), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        parts = [f"SELECT inchi_key FROM {t}"
                 for t in ("chembl_molecules", "pubchem_compounds") if t in tables]
        if not parts:
            return []
        sql = " UNION ".join(parts)
        rows = con.execute(
            f"SELECT DISTINCT inchi_key FROM ({sql}) t WHERE inchi_key IS NOT NULL"
        ).fetchall()
        return [r[0] for r in rows if r[0]]
    finally:
        con.close()


def _record(row: dict) -> dict:
    cpc = row.get("cpc") or []
    assignee = row.get("assignee") or []
    pub = row.get("publication_date")
    return {
        "patent_number": row.get("patent_number"),
        "country": row.get("country"),
        "title": row.get("title"),
        "pub_year": str(pub)[:4] if pub else None,
        "assignees": "; ".join(a for a in assignee if a) or None,
        "keywords": "; ".join(c for c in cpc if c) or None,
    }


def search(inchikeys: list[str], limit: int = MAX_PATENTS) -> list[dict]:
    """Recent patents (up to `limit`) disclosing any of `inchikeys`. Keyless."""
    if not inchikeys:
        return []
    con = duckdb.connect()
    try:
        con.execute("SET memory_limit='2GB'; SET threads=2; INSTALL httpfs; LOAD httpfs;")
        keys = ",".join("'" + k.replace("'", "") + "'" for k in inchikeys)
        sql = f"""
          WITH cmp AS (
            SELECT id FROM read_parquet('{BASE}/compounds.parquet')
            WHERE inchi_key IN ({keys})
          ),
          pats AS (
            SELECT DISTINCT patent_id
            FROM read_parquet('{BASE}/patent_compound_map.parquet')
            WHERE compound_id IN (SELECT id FROM cmp)
          )
          SELECT patent_number, country, title, publication_date, cpc, assignee
          FROM read_parquet('{BASE}/patents.parquet')
          WHERE id IN (SELECT patent_id FROM pats)
          ORDER BY publication_date DESC NULLS LAST
          LIMIT {int(limit)}
        """
        cols = ["patent_number", "country", "title", "publication_date", "cpc", "assignee"]
        return [dict(zip(cols, row, strict=False)) for row in con.execute(sql).fetchall()]
    finally:
        con.close()


def parse(raw: str) -> dict:
    """Parse a stored SureChEMBL patent (JSON) into {'meta', 'sections'}."""
    try:
        r = json.loads(raw)
    except json.JSONDecodeError:
        return {"meta": {}, "sections": []}
    sections = []
    if r.get("title"):
        sections.append({"sec_type": "title", "sec_title": None, "text": r["title"]})
    return {"meta": r, "sections": sections}


def ingest(query: str = "", limit: int = MAX_PATENTS) -> Path:
    """Land SureChEMBL patents for our tracked compounds (keyless).

    `query` may be a single InChIKey to target one compound; empty -> all warehouse
    compounds. Returns the landing dir.
    """
    src_dir = config.raw_source_dir("surechembl")
    manifest = src_dir / "manifest.jsonl"
    keys = [query] if (query and _INCHIKEY.match(query)) else _warehouse_inchikeys()
    if not keys:
        print("[ingest]  surechembl: no InChIKeys to query "
              "(harvest chembl/pubchem compounds first). Skipping.")
        return src_dir
    cap = int(limit) if query else max(int(limit), MAX_PATENTS)
    try:
        records = search(keys, limit=cap)
    except Exception as e:  # noqa: BLE001 - network/parquet failure shouldn't kill the harvest
        print(f"[ingest]  surechembl: query failed ({type(e).__name__}: {e}). Skipping.")
        return src_dir
    print(f"[ingest]  surechembl: {len(records)} patents for {len(keys)} compounds")

    fetched_at = datetime.now(timezone.utc).isoformat()
    built = []
    for raw in records:
        r = _record(raw)
        pn = r["patent_number"]
        if not pn:
            continue
        pid = f"{r['country'] or ''}{pn}"
        json_path = src_dir / f"SCHEMBL_{pid}.json"
        json_path.write_text(json.dumps(r), encoding="utf-8")
        built.append({
            "pmcid": f"patent:{pid}", "pmid": None, "doi": None,
            "title": r["title"], "journal": r["assignees"], "pub_year": r["pub_year"],
            "authors": r["assignees"], "source": "surechembl",
            "query": query or "warehouse", "fetched_at": fetched_at,
            "xml_file": config.rel_data_path(json_path), "has_body": False,
            "abstract": None, "mesh": None, "keywords": r["keywords"],
            "grants": None, "cited_by": None,
        })
    total, added = merge_jsonl(manifest, built, "pmcid")
    print(f"[ingest]  manifest +{added} new ({total} total) -> {manifest.relative_to(config.ROOT)}")
    return src_dir
