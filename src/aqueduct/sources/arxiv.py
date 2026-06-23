"""arXiv connector (math / CS / physics / quant-bio / stats — metadata + full text).

arXiv's API returns Atom XML with rich metadata and the abstract. Optional full body
text is extracted from the PDF (`--fulltext`, needs `pip install -e '.[pdf]'`) and
injected into the stored entry so the document pipeline produces full-text sections.
"""

from __future__ import annotations

import io
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from .. import config, net
from ..landing import merge_jsonl

PDF_URL = "https://arxiv.org/pdf/{}.pdf"
_FT_TAG = f"{{{'http://arxiv.org/schemas/atom'}}}fulltext"  # arxiv-ns child element

API = "https://export.arxiv.org/api/query"
USER_AGENT = "aqueduct/0.1 (data pipeline)"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
PAGE_DELAY = 3.0  # arXiv asks ~3s between requests

ET.register_namespace("", NS["a"])
ET.register_namespace("arxiv", NS["arxiv"])


def _get(url: str, *, retries: int = 3, timeout: int = 30) -> bytes:
    """HTTP GET via the shared resilient client (retry/backoff/rate-limit/breaker)."""
    return net.get_bytes(url, timeout=timeout, retries=retries)


def _build_query(query: str, categories: list[str] | None) -> str:
    if categories:
        cats = " OR ".join(f"cat:{c}" for c in categories)
        return f"({cats}) AND (all:{query})"
    return f"all:{query}"


def search(query: str, limit: int = 25, categories: list[str] | None = None,
           cursor: str | None = None) -> tuple[list[ET.Element], str | None]:
    """Return ``(entries, next_cursor)`` — up to `limit` arXiv <entry> elements.

    `cursor` is the result offset to start from (stringified int); a previous run
    leaves the next offset here so successive harvests page deeper into the history
    rather than re-reading the newest page. `next_cursor` is None at end-of-results,
    signalling the caller to restart from offset 0 next cycle (catching new submissions).
    """
    entries: list[ET.Element] = []
    search_q = _build_query(query, categories)
    start = int(cursor) if cursor else 0
    next_cursor: str | None = None
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
            next_cursor = None  # ran off the end -> restart from the top next cycle
            break
        entries.extend(page_entries)
        start += len(page_entries)
        next_cursor = str(start)
        if len(page_entries) < page:
            next_cursor = None  # last (partial) page -> restart next cycle
            break
        time.sleep(PAGE_DELAY)
    return entries[:limit], next_cursor


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


def _cached_pdf(arxiv_id: str) -> bytes:
    """Return PDF bytes for an arXiv id, downloading once and caching on disk.

    arXiv asks not to re-download the same PDF; the cache makes re-harvests free and
    lets a re-run reuse prior downloads instead of hammering the server.
    """
    cache = config.cache_dir("arxiv_pdf") / f"{arxiv_id.replace('/', '_')}.pdf"
    if cache.exists() and cache.stat().st_size > 0:
        return cache.read_bytes()
    data = _get(PDF_URL.format(arxiv_id), timeout=60)
    cache.write_bytes(data)
    return data


def extract_pdf_text(arxiv_id: str) -> str | None:
    """Download (or reuse a cached) arXiv PDF and extract its text (needs 'pdf' extra)."""
    try:
        from pdfminer.high_level import extract_text
    except ImportError:  # pragma: no cover - environment dependent
        raise RuntimeError("full text needs pdfminer.six: pip install -e '.[pdf]'") from None
    try:
        data = _cached_pdf(arxiv_id)
        text = extract_text(io.BytesIO(data)) or ""
    except Exception:  # noqa: BLE001 - a bad/again PDF shouldn't kill the batch
        return None
    # strip control chars that are invalid in XML 1.0 (PDFs emit form-feeds etc.),
    # else the injected text breaks re-parsing of the stored entry
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or None


def inject_fulltext(entry: ET.Element, text: str) -> None:
    """Attach extracted full text to an Atom entry (arxiv-namespaced child)."""
    el = entry.find(_FT_TAG)
    if el is None:
        el = ET.SubElement(entry, _FT_TAG)
    el.text = text


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
    ft = entry.find(_FT_TAG)
    if ft is not None and ft.text:
        sections.append({"sec_type": "body", "sec_title": "full text", "text": ft.text})
    return {"meta": m, "sections": sections}


def ingest(query: str, limit: int = 25, categories: list[str] | None = None,
           fulltext: bool = False, cursor: str | None = None) -> tuple[Path, str | None]:
    """Land arXiv entries (Atom XML) + a metadata manifest in the landing zone.

    With `fulltext=True`, the PDF is downloaded and its text injected into the stored
    entry so the document pipeline produces full-text sections (else abstract-only).
    Resumes paging from `cursor` and returns ``(landing_dir, next_cursor)``.
    """
    src_dir = config.raw_source_dir("arxiv")
    manifest = src_dir / "manifest.jsonl"
    entries, next_cursor = search(query, limit=limit, categories=categories, cursor=cursor)
    print(f"[ingest]  arxiv: {len(entries)} hits for {query!r}{' (+full text)' if fulltext else ''}")

    fetched_at = datetime.now(timezone.utc).isoformat()
    built = []
    for i, entry in enumerate(entries, 1):
        m = _entry_meta(entry)
        doc_id = m["arxiv_id"].replace("/", "_")
        has_body = False
        if fulltext:
            body = extract_pdf_text(m["arxiv_id"])
            if body:
                inject_fulltext(entry, body)
                has_body = True
            time.sleep(PAGE_DELAY)  # be polite between PDF downloads
        xml_path = src_dir / f"{doc_id}.xml"
        xml_path.write_text(ET.tostring(entry, encoding="unicode"), encoding="utf-8")
        built.append({
            # map to the shared document schema (pmcid slot holds the doc id)
            "pmcid": f"arXiv:{m['arxiv_id']}", "pmid": None, "doi": m["doi"],
            "title": m["title"], "journal": m["journal"], "pub_year": m["pub_year"],
            "authors": m["authors"], "source": "arxiv", "query": query,
            "fetched_at": fetched_at, "xml_file": config.rel_data_path(xml_path),
            "has_body": has_body,
            "abstract": m["abstract"], "mesh": None,
            "keywords": m["categories"], "grants": None, "cited_by": None,
        })
        tag = "full-text" if has_body else "abstract"
        print(f"  [{i}/{len(entries)}] arXiv:{m['arxiv_id']} ({tag}) -> {xml_path.name}")
    total, added = merge_jsonl(manifest, built, "pmcid")
    print(f"[ingest]  manifest +{added} new ({total} total) -> {manifest.relative_to(config.ROOT)}")
    return src_dir, next_cursor
