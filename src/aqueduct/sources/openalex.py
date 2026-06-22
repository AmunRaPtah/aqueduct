"""OpenAlex connector (ALL research fields — keyless).

OpenAlex indexes every discipline (math, physics, CS, social science, …) and every
preprint server, so it broadens Aqueduct beyond biomedicine in one connector. It
returns rich metadata + a reconstructable abstract (no API key; a `mailto` joins the
polite pool). Lands as a document so it flows through the corpus pipeline.
"""

from __future__ import annotations

import json
import os
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from .. import config, net
from ..landing import merge_jsonl

API = "https://api.openalex.org/works"
USER_AGENT = "aqueduct/0.1 (data pipeline)"


def _get(url: str, *, retries: int = 3, timeout: int = 30) -> dict:
    """Fetch JSON via the shared resilient client (retry/backoff/rate-limit/breaker)."""
    return net.get_json(url, timeout=timeout, retries=retries)


def reconstruct_abstract(inv: dict | None) -> str | None:
    """Rebuild abstract text from OpenAlex's inverted index {word: [positions]}."""
    if not inv:
        return None
    at: dict[int, str] = {}
    for word, positions in inv.items():
        for p in positions:
            at[p] = word
    return " ".join(at[i] for i in sorted(at)) or None


def _short_id(work_id: str) -> str:
    return (work_id or "").rstrip("/").split("/")[-1]  # https://openalex.org/W123 -> W123


def _record(w: dict) -> dict:
    """Trim an OpenAlex work to the fields we store (kept as the raw doc blob)."""
    loc = w.get("primary_location") or {}
    source = (loc.get("source") or {}) if loc else {}
    authors = [a.get("author", {}).get("display_name")
               for a in w.get("authorships", [])]
    concepts = [c.get("display_name") for c in w.get("concepts", [])]
    return {
        "id": _short_id(w.get("id", "")),
        "doi": (w.get("doi") or "").replace("https://doi.org/", "") or None,
        "title": w.get("title"),
        "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
        "journal": source.get("display_name"),
        "pub_year": w.get("publication_year"),
        "type": w.get("type"),
        "authors": ", ".join(a for a in authors if a),
        "concepts": "; ".join(c for c in concepts if c),
        "cited_by": w.get("cited_by_count"),
    }


def search(query: str, limit: int = 25,
           cursor: str | None = None) -> tuple[list[dict], str | None]:
    """Search OpenAlex works across all fields.

    Returns ``(records, next_cursor)``: trimmed records starting from `cursor`, plus
    the cursor to resume from on the next run — so successive harvests page deeper
    rather than re-reading the same first page. `next_cursor` is None at end-of-results,
    signalling the caller to resweep from the top next cycle.
    """
    out: list[dict] = []
    mark = cursor or "*"
    next_cursor: str | None = None
    mailto = os.environ.get("OPENALEX_MAILTO", "")
    while len(out) < limit:
        per = min(200, limit - len(out))
        params = {"search": query, "per-page": per, "cursor": mark}
        if mailto:
            params["mailto"] = mailto
        data = _get(f"{API}?{urllib.parse.urlencode(params)}")
        results = data.get("results", [])
        if not results:
            next_cursor = None  # exhausted -> resweep from the top next cycle
            break
        out.extend(_record(w) for w in results)
        mark = data.get("meta", {}).get("next_cursor")
        next_cursor = mark
        if not mark:
            break
    return out[:limit], next_cursor


def parse(raw: str) -> dict:
    """Parse a stored OpenAlex work (JSON) into {'meta', 'sections'}."""
    try:
        r = json.loads(raw)
    except json.JSONDecodeError:
        return {"meta": {}, "sections": []}
    sections = []
    if r.get("title"):
        sections.append({"sec_type": "title", "sec_title": None, "text": r["title"]})
    if r.get("abstract"):
        sections.append({"sec_type": "abstract", "sec_title": "abstract", "text": r["abstract"]})
    return {"meta": r, "sections": sections}


def ingest(query: str, limit: int = 25,
           cursor: str | None = None) -> tuple[Path, str | None]:
    """Land OpenAlex works (JSON) + a metadata manifest in the landing zone.

    Resumes paging from `cursor` and returns ``(landing_dir, next_cursor)``.
    """
    src_dir = config.raw_source_dir("openalex")
    manifest = src_dir / "manifest.jsonl"
    records, next_cursor = search(query, limit=limit, cursor=cursor)
    print(f"[ingest]  openalex: {len(records)} works for {query!r}")

    fetched_at = datetime.now(timezone.utc).isoformat()
    built = []
    for r in records:
        doc_id = r["id"] or r.get("doi") or ""
        json_path = src_dir / f"{doc_id}.json"
        json_path.write_text(json.dumps(r), encoding="utf-8")
        built.append({
            "pmcid": f"openalex:{doc_id}", "pmid": None, "doi": r["doi"],
            "title": r["title"], "journal": r["journal"], "pub_year": r["pub_year"],
            "authors": r["authors"], "source": "openalex", "query": query,
            "fetched_at": fetched_at, "xml_file": str(json_path),
            "has_body": False, "abstract": r["abstract"], "mesh": None,
            "keywords": r["concepts"], "grants": None, "cited_by": r["cited_by"],
        })
    total, added = merge_jsonl(manifest, built, "pmcid")
    print(f"[ingest]  manifest +{added} new ({total} total) -> {manifest.relative_to(config.ROOT)}")
    return src_dir, next_cursor
