"""Europe PMC connector.

Discovery via the Europe PMC REST search API (rich query syntax, no key), then
full-text JATS-XML via NCBI E-utilities `efetch` (Europe PMC's own XML route does
not serve these reliably). Both are keyless.

Politeness: NCBI asks for <= 3 requests/sec without an API key and a `tool`/`email`
identifier. Set the env var `NCBI_EMAIL` to identify your traffic.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from .. import config, net
from ..landing import merge_jsonl

SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
USER_AGENT = "aqueduct/0.1 (+https://github.com/; data pipeline)"
EFETCH_DELAY = 0.34  # seconds between efetch calls -> <= 3 req/s


def _get(url: str, *, retries: int = 3, timeout: int = 30) -> bytes:
    """HTTP GET via the shared resilient client (retry/backoff/rate-limit/breaker)."""
    return net.get_bytes(url, timeout=timeout, retries=retries)


def search(query: str, limit: int = 25,
           cursor: str | None = None) -> tuple[list[dict], str | None]:
    """Search Europe PMC for open-access, in-EPMC, full-text articles.

    Returns ``(records, next_cursor)``: up to `limit` metadata records (newest first),
    starting from `cursor` (the cursorMark a previous run left off at), plus the
    cursorMark to resume from next time. This makes successive harvests paginate
    *deeper* into the result set instead of re-reading the same newest page every run.
    `next_cursor` is None when the result set is exhausted — the caller resets to the
    top on the next cycle to pick up newly published papers.
    """
    full_query = f"({query}) AND OPEN_ACCESS:y AND IN_EPMC:y AND HAS_FT:y"
    out: list[dict] = []
    mark = cursor or "*"
    next_cursor: str | None = None
    while len(out) < limit:
        page_size = min(100, limit - len(out))
        params = urllib.parse.urlencode(
            {
                "query": full_query,
                "format": "json",
                "resultType": "core",  # rich metadata: MeSH, keywords, grants, citations
                "pageSize": page_size,
                "cursorMark": mark,
                "sort": "P_PDATE_D desc",
            }
        )
        data = json.loads(_get(f"{SEARCH_URL}?{params}"))
        results = data.get("resultList", {}).get("result", [])
        if not results:
            next_cursor = None  # end of results -> resweep from the top next cycle
            break
        for r in results:
            pmcid = r.get("pmcid")
            if not pmcid:  # skip records without a PMC full-text id
                continue
            mesh = [m.get("descriptorName") for m in
                    r.get("meshHeadingList", {}).get("meshHeading", [])]
            keywords = r.get("keywordList", {}).get("keyword", [])
            grants = [g.get("agency") for g in
                      r.get("grantsList", {}).get("grant", [])]
            out.append(
                {
                    "pmcid": pmcid,
                    "pmid": r.get("pmid"),
                    "doi": r.get("doi"),
                    "title": r.get("title"),
                    # core nests the journal under journalInfo; lite used journalTitle
                    "journal": r.get("journalTitle")
                    or r.get("journalInfo", {}).get("journal", {}).get("title"),
                    "pub_year": r.get("pubYear"),
                    "authors": r.get("authorString"),
                    "abstract": r.get("abstractText"),
                    "mesh": "; ".join(m for m in mesh if m) or None,
                    "keywords": "; ".join(k for k in keywords if k) or None,
                    "grants": "; ".join(sorted({g for g in grants if g})) or None,
                    "cited_by": r.get("citedByCount"),
                }
            )
            if len(out) >= limit:
                break
        new_mark = data.get("nextCursorMark")
        if not new_mark or new_mark == mark:
            next_cursor = None  # reached the end of the result set
            break
        mark = new_mark
        next_cursor = mark  # a valid point to resume from on the next run
    return out, next_cursor


def fetch_fulltext_xml(pmcid: str) -> str:
    """Fetch JATS-XML full text for a PMCID (e.g. 'PMC4564304') via efetch."""
    numeric = pmcid.removeprefix("PMC")
    params = {"db": "pmc", "id": numeric, "rettype": "xml", "retmode": "xml", "tool": "aqueduct"}
    email = os.environ.get("NCBI_EMAIL")
    if email:
        params["email"] = email
    return _get(f"{EFETCH_URL}?{urllib.parse.urlencode(params)}").decode("utf-8", "replace")


def ingest(query: str, limit: int = 25,
           cursor: str | None = None) -> tuple[Path, str | None]:
    """Land raw full-text XML + a metadata manifest in the bronze landing zone.

    Writes one `<pmcid>.xml` per article and appends a record per article to
    `manifest.jsonl`. Resumes paging from `cursor` and returns
    ``(landing_dir, next_cursor)`` so the harvester can persist where to continue.
    """
    src_dir = config.raw_source_dir("europepmc")
    manifest = src_dir / "manifest.jsonl"
    records, next_cursor = search(query, limit=limit, cursor=cursor)
    print(f"[ingest]  europepmc: {len(records)} hits for {query!r}")

    fetched_at = datetime.now(timezone.utc).isoformat()
    built = []
    for i, rec in enumerate(records, 1):
        pmcid = rec["pmcid"]
        try:
            xml = fetch_fulltext_xml(pmcid)
        except Exception as e:  # noqa: BLE001 - skip a bad doc, keep the batch
            print(f"  ! {pmcid}: fetch failed ({e})")
            continue
        xml_path = src_dir / f"{pmcid}.xml"
        xml_path.write_text(xml, encoding="utf-8")
        has_body = "<body>" in xml
        built.append({
            **rec, "source": "europepmc", "query": query, "fetched_at": fetched_at,
            "xml_file": str(xml_path), "has_body": has_body,
        })
        print(f"  [{i}/{len(records)}] {pmcid} {'full-text' if has_body else 'abstract-only'} -> {xml_path.name}")
        time.sleep(EFETCH_DELAY)
    total, added = merge_jsonl(manifest, built, "pmcid")
    print(f"[ingest]  manifest +{added} new ({total} total) -> {manifest.relative_to(config.ROOT)}")
    return src_dir, next_cursor
