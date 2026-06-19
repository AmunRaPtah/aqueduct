"""arXiv connector (math / CS / physics / quant-bio / stats — full metadata + abstract).

arXiv's API returns Atom XML with rich metadata and the abstract. Full body text
lives only in the PDF / LaTeX e-print (a later systematic increment); for now each
article lands as title + abstract sections, with the PDF URL kept in metadata.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from .. import config

API = "https://export.arxiv.org/api/query"
USER_AGENT = "aqueduct/0.1 (data pipeline)"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
PAGE_DELAY = 3.0  # arXiv asks ~3s between requests

ET.register_namespace("", NS["a"])
ET.register_namespace("arxiv", NS["arxiv"])


def _get(url: str, *, retries: int = 3, timeout: int = 30) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url}") from last


def _build_query(query: str, categories: list[str] | None) -> str:
    if categories:
        cats = " OR ".join(f"cat:{c}" for c in categories)
        return f"({cats}) AND (all:{query})"
    return f"all:{query}"


def search(query: str, limit: int = 25, categories: list[str] | None = None) -> list[ET.Element]:
    """Return up to `limit` arXiv <entry> elements (newest first)."""
    entries: list[ET.Element] = []
    search_q = _build_query(query, categories)
    start = 0
    while len(entries) < limit:
        page = min(100, limit - len(entries))
        params = urllib.parse.urlencode(
            {
                "search_query": search_q,
                "start": start,
                "max_results": page,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        feed = ET.fromstring(_get(f"{API}?{params}"))
        page_entries = feed.findall("a:entry", NS)
        if not page_entries:
            break
        entries.extend(page_entries)
        start += len(page_entries)
        if len(page_entries) < page:
            break
        time.sleep(PAGE_DELAY)
    return entries[:limit]


def _entry_meta(entry: ET.Element) -> dict:
    def text(path: str) -> str | None:
        el = entry.find(path, NS)
        return el.text.strip() if el is not None and el.text else None

    arxiv_url = text("a:id") or ""
    arxiv_id = arxiv_url.rsplit("/", 1)[-1]
    authors = [a.text.strip() for a in entry.findall("a:author/a:name", NS) if a.text]
    primary = entry.find("arxiv:primary_category", NS)
    cats = [c.get("term") for c in entry.findall("a:category", NS)]
    pdf = next(
        (l.get("href") for l in entry.findall("a:link", NS) if l.get("title") == "pdf"),
        None,
    )
    published = text("a:published")
    return {
        "arxiv_id": arxiv_id,
        "title": " ".join((text("a:title") or "").split()),
        "abstract": " ".join((text("a:summary") or "").split()),
        "authors": ", ".join(authors),
        "doi": text("arxiv:doi"),
        "journal": text("arxiv:journal_ref"),
        "primary_category": primary.get("term") if primary is not None else None,
        "categories": ",".join(c for c in cats if c),
        "pub_year": published[:4] if published else None,
        "published": published,
        "pdf_url": pdf,
    }


def parse_atom(xml: str) -> dict:
    """Parse a stored arXiv Atom entry into {'meta', 'sections'} (document pipeline)."""
    try:
        entry = ET.fromstring(xml)
    except ET.ParseError:
        return {"meta": {}, "sections": []}
    m = _entry_meta(entry)
    sections = []
    if m["title"]:
        sections.append({"sec_type": "title", "sec_title": None, "text": m["title"]})
    if m["abstract"]:
        sections.append({"sec_type": "abstract", "sec_title": "abstract", "text": m["abstract"]})
    return {"meta": m, "sections": sections}


def ingest(query: str, limit: int = 25, categories: list[str] | None = None) -> Path:
    """Land arXiv entries (Atom XML) + a metadata manifest in the landing zone."""
    src_dir = config.raw_source_dir("arxiv")
    manifest = src_dir / "manifest.jsonl"
    entries = search(query, limit=limit, categories=categories)
    print(f"[ingest]  arxiv: {len(entries)} hits for {query!r}")

    fetched_at = datetime.now(timezone.utc).isoformat()
    with manifest.open("w") as mf:
        for i, entry in enumerate(entries, 1):
            m = _entry_meta(entry)
            doc_id = m["arxiv_id"].replace("/", "_")
            xml_path = src_dir / f"{doc_id}.xml"
            xml_path.write_text(ET.tostring(entry, encoding="unicode"), encoding="utf-8")
            rec = {
                # map to the shared document schema (pmcid slot holds the doc id)
                "pmcid": f"arXiv:{m['arxiv_id']}",
                "pmid": None,
                "doi": m["doi"],
                "title": m["title"],
                "journal": m["journal"],
                "pub_year": m["pub_year"],
                "authors": m["authors"],
                "source": "arxiv",
                "query": query,
                "fetched_at": fetched_at,
                "xml_file": str(xml_path),
                "has_body": False,  # abstract-only until PDF/LaTeX extraction
                "abstract": m["abstract"],
                "mesh": None,  # arXiv has no MeSH
                "keywords": m["categories"],  # arXiv subject categories
                "grants": None,
                "cited_by": None,
            }
            mf.write(json.dumps(rec) + "\n")
            print(f"  [{i}/{len(entries)}] {rec['pmcid']} -> {xml_path.name}")
    print(f"[ingest]  manifest -> {manifest.relative_to(config.ROOT)}")
    return src_dir
