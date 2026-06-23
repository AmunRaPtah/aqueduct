"""Patents connector (USPTO via PatentsView Search API).

Patents span every field — chemistry, devices, ML, materials — so this widens
Aqueduct beyond academic papers. The modern PatentsView API needs a free API key:
register at https://patentsview.org and set `PATENTSVIEW_API_KEY`. Without it the
connector no-ops with a clear message. Lands as a document (title + abstract).
"""

from __future__ import annotations

import json
import os
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from .. import config, net
from ..landing import merge_jsonl

API = "https://search.patentsview.org/api/v1/patent/"
USER_AGENT = "aqueduct/0.1 (data pipeline)"
FIELDS = [
    "patent_id", "patent_title", "patent_abstract", "patent_date",
    "assignees.assignee_organization", "cpc_current.cpc_group_id",
]


def _get(url: str, key: str, *, timeout: int = 30) -> dict | None:
    """Fetch JSON with the API key header, returning None on any network failure."""
    try:
        return net.get_json(url, timeout=timeout, retries=1, headers={"X-Api-Key": key})
    except net.NetworkError:
        return None


def _record(p: dict) -> dict:
    assignees = [a.get("assignee_organization") for a in (p.get("assignees") or [])]
    cpc = [c.get("cpc_group_id") for c in (p.get("cpc_current") or [])]
    date = p.get("patent_date") or ""
    keywords = [k for k in assignees + cpc if k]
    return {
        "patent_id": p.get("patent_id"),
        "title": p.get("patent_title"),
        "abstract": p.get("patent_abstract"),
        "pub_year": date[:4] or None,
        "assignees": "; ".join(a for a in assignees if a) or None,
        "keywords": "; ".join(keywords) or None,
    }


def search(query: str, limit: int = 25, key: str | None = None,
           cursor: str | None = None) -> tuple[list[dict], str | None]:
    """Text-search US patents by title/abstract. Needs an API key.

    Returns ``(records, next_cursor)``. Pages with PatentsView's `after` cursor over a
    stable `patent_id` sort, so each run resumes after the last patent the previous run
    saw (paging deeper) rather than re-reading the same page. `next_cursor` is None at
    end-of-results, signalling the caller to restart from the top next cycle.
    """
    key = key or os.environ.get("PATENTSVIEW_API_KEY")
    if not key:
        return [], None
    q = {"_or": [{"_text_any": {"patent_title": query}},
                 {"_text_any": {"patent_abstract": query}}]}
    size = min(limit, 1000)
    opts: dict = {"size": size}
    if cursor:
        opts["after"] = [cursor]  # resume after the prior run's last patent_id
    params = urllib.parse.urlencode({
        "q": json.dumps(q),
        "f": json.dumps(FIELDS),
        "s": json.dumps([{"patent_id": "asc"}]),  # stable sort enables cursor paging
        "o": json.dumps(opts),
    })
    data = _get(f"{API}?{params}", key)
    if not data:
        return [], None
    raw = data.get("patents", [])
    records = [_record(p) for p in raw][:limit]
    # only more to fetch if the page came back full; else reset to the top next cycle
    next_cursor = raw[-1].get("patent_id") if raw and len(raw) >= size else None
    return records, next_cursor


def parse(raw: str) -> dict:
    """Parse a stored patent (JSON) into {'meta', 'sections'}."""
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
    """Land US patents (JSON) + a metadata manifest in the landing zone.

    Resumes paging from `cursor` and returns ``(landing_dir, next_cursor)``.
    """
    src_dir = config.raw_source_dir("patents")
    manifest = src_dir / "manifest.jsonl"
    if not os.environ.get("PATENTSVIEW_API_KEY"):
        print("[ingest]  patents: set PATENTSVIEW_API_KEY (free at patentsview.org) to enable. Skipping.")
        return src_dir, None
    records, next_cursor = search(query, limit=limit, cursor=cursor)
    print(f"[ingest]  patents: {len(records)} patents for {query!r}")

    fetched_at = datetime.now(timezone.utc).isoformat()
    built = []
    for r in records:
        pid = r["patent_id"]
        json_path = src_dir / f"US{pid}.json"
        json_path.write_text(json.dumps(r), encoding="utf-8")
        built.append({
            "pmcid": f"patent:US{pid}", "pmid": None, "doi": None,
            "title": r["title"], "journal": r["assignees"], "pub_year": r["pub_year"],
            "authors": r["assignees"], "source": "patents", "query": query,
            "fetched_at": fetched_at, "xml_file": config.rel_data_path(json_path),
            "has_body": False, "abstract": r["abstract"], "mesh": None,
            "keywords": r["keywords"], "grants": None, "cited_by": None,
        })
    total, added = merge_jsonl(manifest, built, "pmcid")
    print(f"[ingest]  manifest +{added} new ({total} total) -> {manifest.relative_to(config.ROOT)}")
    return src_dir, next_cursor
