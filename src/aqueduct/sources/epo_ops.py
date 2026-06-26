"""EPO Open Patent Services (OPS) connector — worldwide patents, free tier.

Free, but needs OAuth2 client-credentials: register a non-paying app at
https://developers.epo.org and set EPO_OPS_KEY + EPO_OPS_SECRET. Without them this
connector no-ops with a clear message. Fair-use quota is ~4 GB/week. Text-searches the
published-data biblio service and lands each hit (publication number + title) as a
document.

NOTE: this connector is credential-gated and has not been verified against the live OPS
service from this environment (no creds available here). The request/JSON-extraction
paths are defensive; verify once EPO_OPS_KEY/SECRET are set.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from .. import config, net
from ..landing import merge_jsonl

AUTH = "https://ops.epo.org/3.2/auth/accesstoken"
SEARCH = "https://ops.epo.org/3.2/rest-services/published-data/search/biblio"
PAGE = 25  # OPS allows up to 100 results per range request


def _creds() -> tuple[str | None, str | None]:
    return os.environ.get("EPO_OPS_KEY"), os.environ.get("EPO_OPS_SECRET")


def _token(key: str, secret: str) -> str | None:
    cred = base64.b64encode(f"{key}:{secret}".encode()).decode()
    try:
        raw = net.request(
            AUTH, data=b"grant_type=client_credentials",
            headers={"Authorization": f"Basic {cred}",
                     "Content-Type": "application/x-www-form-urlencoded"},
            timeout=30, retries=1)
        return json.loads(raw).get("access_token")
    except (net.NetworkError, json.JSONDecodeError, KeyError):
        return None


def _walk(node, out_ids: list, out_titles: list) -> None:
    """Defensively collect document-id numbers and invention-titles from OPS JSON."""
    if isinstance(node, dict):
        if "document-id" in node and isinstance(node["document-id"], dict):
            d = node["document-id"]
            country = (d.get("country") or {}).get("$", "")
            num = (d.get("doc-number") or {}).get("$", "")
            kind = (d.get("kind") or {}).get("$", "")
            if num:
                out_ids.append((f"{country}{num}{kind}", country or None))
        if "invention-title" in node:
            it = node["invention-title"]
            its = it if isinstance(it, list) else [it]
            for t in its:
                if isinstance(t, dict) and t.get("$"):
                    out_titles.append(t["$"])
        for v in node.values():
            _walk(v, out_ids, out_titles)
    elif isinstance(node, list):
        for v in node:
            _walk(v, out_ids, out_titles)


def search(query: str, limit: int = PAGE, cursor: str | None = None) -> tuple[list[dict], str | None]:
    """Search OPS biblio. Returns (records, next_cursor). Needs creds."""
    key, secret = _creds()
    if not key or not secret:
        return [], None
    token = _token(key, secret)
    if not token:
        return [], None
    begin = int(cursor) if cursor else 1
    end = begin + min(limit, 100) - 1
    params = urllib.parse.urlencode({"q": f'ti="{query}" or ab="{query}"'})
    try:
        raw = net.request(
            f"{SEARCH}?{params}",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json",
                     "X-OPS-Range": f"{begin}-{end}"},
            timeout=45, retries=1)
        data = json.loads(raw)
    except (net.NetworkError, json.JSONDecodeError):
        return [], None
    ids: list = []
    titles: list = []
    _walk(data, ids, titles)
    records = []
    for i, (pid, country) in enumerate(ids):
        records.append({"patent_number": pid, "country": country,
                        "title": titles[i] if i < len(titles) else None})
    next_cursor = str(end + 1) if len(records) >= (end - begin + 1) else None
    return records, next_cursor


def parse(raw: str) -> dict:
    """Parse a stored OPS patent (JSON) into {'meta', 'sections'}."""
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


def ingest(query: str, limit: int = PAGE, cursor: str | None = None) -> tuple[Path, str | None]:
    """Land EPO OPS patents for `query`. Resumes from `cursor`; returns (dir, next_cursor)."""
    src_dir = config.raw_source_dir("epo_ops")
    manifest = src_dir / "manifest.jsonl"
    key, secret = _creds()
    if not key or not secret:
        print("[ingest]  epo_ops: set EPO_OPS_KEY + EPO_OPS_SECRET "
              "(free at developers.epo.org) to enable. Skipping.")
        return src_dir, None
    records, next_cursor = search(query, limit=limit, cursor=cursor)
    print(f"[ingest]  epo_ops: {len(records)} patents for {query!r}")

    fetched_at = datetime.now(timezone.utc).isoformat()
    built = []
    for r in records:
        pid = r.get("patent_number")
        if not pid:
            continue
        json_path = src_dir / f"EPO_{pid}.json"
        json_path.write_text(json.dumps(r), encoding="utf-8")
        built.append({
            "pmcid": f"patent:{pid}", "pmid": None, "doi": None,
            "title": r.get("title"), "journal": None, "pub_year": None,
            "authors": None, "source": "epo_ops", "query": query,
            "fetched_at": fetched_at, "xml_file": config.rel_data_path(json_path),
            "has_body": False, "abstract": r.get("abstract"), "mesh": None,
            "keywords": None, "grants": None, "cited_by": None,
        })
    total, added = merge_jsonl(manifest, built, "pmcid")
    print(f"[ingest]  manifest +{added} new ({total} total) -> {manifest.relative_to(config.ROOT)}")
    return src_dir, next_cursor
