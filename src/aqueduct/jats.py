"""Minimal JATS-XML parser (stdlib only).

Turns an efetch JATS document into metadata + an ordered list of sections
(title / abstract / body). Deliberately simple: it flattens each top-level body
`<sec>` into one text block. Finer structure (nested sections, tables, figures,
references, citations) is left for systematic later passes.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

_WS = re.compile(r"\s+")


def _text(el: ET.Element | None) -> str:
    """All descendant text of an element, whitespace-normalised."""
    if el is None:
        return ""
    return _WS.sub(" ", "".join(el.itertext())).strip()


def _first(root: ET.Element, path: str) -> ET.Element | None:
    return root.find(path)


def _article(root: ET.Element) -> ET.Element:
    """Return the <article> element whether or not it's wrapped in a set."""
    if root.tag == "article":
        return root
    found = root.find(".//article")
    return found if found is not None else root


def _meta(article: ET.Element) -> dict:
    def id_of(kind: str) -> str | None:
        for el in article.findall(".//front//article-id"):
            if el.get("pub-id-type") == kind:
                return (el.text or "").strip() or None
        return None

    authors = []
    for c in article.findall(".//front//contrib[@contrib-type='author']"):
        sur = _text(c.find(".//surname"))
        giv = _text(c.find(".//given-names"))
        name = " ".join(p for p in (giv, sur) if p)
        if name:
            authors.append(name)

    year_el = _first(article, ".//front//pub-date/year")
    pmc = id_of("pmc") or id_of("pmcid")

    return {
        "title": _text(_first(article, ".//front//article-title")),
        "journal": _text(_first(article, ".//front//journal-title")),
        "doi": id_of("doi"),
        "pmid": id_of("pmid"),
        "pmcid": f"PMC{pmc}" if pmc and not pmc.startswith("PMC") else pmc,
        "pub_year": _text(year_el) or None,
        "authors": authors,
    }


def _sections(article: ET.Element, meta: dict) -> list[dict]:
    sections: list[dict] = []

    if meta["title"]:
        sections.append({"sec_type": "title", "sec_title": None, "text": meta["title"],
                         "n_figures": 0, "n_tables": 0})

    for abs in article.findall(".//front//abstract"):
        txt = _text(abs)
        if txt:
            label = abs.get("abstract-type") or "abstract"
            sections.append({"sec_type": "abstract", "sec_title": label, "text": txt,
                             "n_figures": 0, "n_tables": 0})

    body = _first(article, ".//body")
    if body is not None:
        # Emit one section per <sec> using its DIRECT <p> children, so a section's
        # text is its own prose, not its subsections' (those become their own rows).
        # sec_title carries the full heading path, e.g. "1. Introduction > 1.1 ...".
        def walk(el: ET.Element, trail: list[str]) -> None:
            for sec in el.findall("sec"):
                title = _text(sec.find("title")) or None
                path = trail + ([title] if title else [])
                direct = _WS.sub(" ", " ".join(_text(p) for p in sec.findall("p"))).strip()
                if direct:
                    # figures/tables attached directly to this <sec> (subsecs count their own)
                    sections.append(
                        {
                            "sec_type": "body",
                            "sec_title": " > ".join(path) or None,
                            "text": direct,
                            "n_figures": len(sec.findall("fig")),
                            "n_tables": len(sec.findall("table-wrap")),
                        }
                    )
                walk(sec, path)

        # paragraphs sitting directly under <body> with no enclosing <sec>
        stray = _WS.sub(" ", " ".join(_text(p) for p in body.findall("p"))).strip()
        if stray:
            sections.append({"sec_type": "body", "sec_title": None, "text": stray,
                             "n_figures": len(body.findall("fig")),
                             "n_tables": len(body.findall("table-wrap"))})
        walk(body, [])

    return sections


def parse_jats(xml: str) -> dict:
    """Parse a JATS-XML string into {'meta': {...}, 'sections': [...]}.

    Returns sections=[] (with whatever meta parsed) if the body is absent or the
    XML can't be parsed, so a single bad document never breaks a batch.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return {"meta": {}, "sections": []}
    article = _article(root)
    meta = _meta(article)
    return {"meta": meta, "sections": _sections(article, meta)}
