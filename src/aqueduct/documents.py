"""Document pipeline for full-text corpora (bronze -> silver -> gold).

  store_documents   landing-zone XML + manifest  -> documents_raw   (bronze)
  process_documents documents_raw                -> doc_sections    (silver)
  chunk_documents   doc_sections                 -> doc_chunks      (gold)

Chunks are search/embedding-ready: bounded word windows that never cross a
section boundary, each carrying its source section heading.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import duckdb

from . import config, jats
from .sources import (arxiv, epo_ops, google_patents, openalex, patents,
                      surechembl)
from .storage import connect

# Chunking is configurable via env (so a deploy can tune retrieval granularity
# without code changes); these are the defaults.
CHUNK_WORDS = int(os.environ.get("AQUEDUCT_CHUNK_WORDS", "220"))      # target words/chunk
CHUNK_OVERLAP = int(os.environ.get("AQUEDUCT_CHUNK_OVERLAP", "40"))   # words shared between chunks
CHUNK_SENTENCE_AWARE = os.environ.get("AQUEDUCT_CHUNK_SENTENCE_AWARE", "1") != "0"

# --- IMRaD section classification -----------------------------------------
# Map a section heading to a canonical kind so retrieval can filter by structure
# (e.g. "give me Methods/Results only"). Ordered: first keyword hit wins.
_SECTION_KINDS = [
    ("methods", ("method", "materials and methods", "experimental", "procedure",
                 "data collection", "statistical analys", "study design")),
    ("results", ("result", "findings")),
    ("discussion", ("discussion",)),
    ("conclusion", ("conclusion", "concluding", "summary")),
    ("introduction", ("introduction", "background")),
    ("related", ("related work", "literature review", "prior work")),
    ("limitations", ("limitation",)),
]


def section_kind(sec_type: str, sec_title: str | None) -> str:
    """Canonical IMRaD-ish kind from a section's type + heading.

    title/abstract pass through; body headings are matched against keyword sets;
    anything unrecognised stays generic 'body'.
    """
    if sec_type in ("title", "abstract"):
        return sec_type
    hay = (sec_title or "").lower()
    for kind, keys in _SECTION_KINDS:
        if any(k in hay for k in keys):
            return kind
    return "body"


_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")  # sentence boundary heuristic

# Each source lands a different raw format; parse it back with the right parser.
PARSERS = {
    "europepmc": jats.parse_jats,   # JATS-XML
    "arxiv": arxiv.parse_atom,      # Atom XML
    "openalex": openalex.parse,     # OpenAlex JSON
    "patents": patents.parse,       # PatentsView JSON
    "surechembl": surechembl.parse,        # SureChEMBL patent JSON
    "epo_ops": epo_ops.parse,              # EPO OPS patent JSON
    "google_patents": google_patents.parse,  # Google Patents (BigQuery) JSON
}


# --------------------------------------------------------------------------- #
# bronze
# --------------------------------------------------------------------------- #
def _iter_manifests() -> list[Path]:
    """All source manifests under the landing zone (one per connector)."""
    return sorted(config.RAW_DIR.glob("*/manifest.jsonl"))


def store_documents(con: duckdb.DuckDBPyConnection | None = None) -> int:
    """Load landing-zone XML + metadata into `documents_raw`. Returns row count."""
    owns = con is None
    con = con or connect()
    try:
        con.execute(
            """
            CREATE OR REPLACE TABLE documents_raw (
                pmcid TEXT PRIMARY KEY, pmid TEXT, doi TEXT, title TEXT,
                journal TEXT, pub_year INTEGER, authors TEXT, source TEXT,
                query TEXT, fetched_at TEXT, has_body BOOLEAN,
                abstract TEXT, mesh TEXT, keywords TEXT, grants TEXT, cited_by INTEGER,
                raw_xml TEXT
            )
            """
        )
        rows = 0
        for manifest in _iter_manifests():
            for line in manifest.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                xml_path = config.resolve_data_path(rec["xml_file"])
                if not xml_path.exists():
                    continue
                year = rec.get("pub_year")
                cited = rec.get("cited_by")
                con.execute(
                    "INSERT OR REPLACE INTO documents_raw VALUES "
                    "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [
                        rec["pmcid"], rec.get("pmid"), rec.get("doi"), rec.get("title"),
                        rec.get("journal"), int(year) if year else None,
                        rec.get("authors"), rec.get("source"), rec.get("query"),
                        rec.get("fetched_at"), rec.get("has_body"),
                        rec.get("abstract"), rec.get("mesh"), rec.get("keywords"),
                        rec.get("grants"), int(cited) if cited is not None else None,
                        xml_path.read_text(encoding="utf-8"),
                    ],
                )
                rows += 1
        total = con.execute("SELECT count(*) FROM documents_raw").fetchone()[0]
        print(f"[store]   {rows} docs loaded -> documents_raw (bronze, {total} total)")
        return rows
    finally:
        if owns:
            con.close()


# --------------------------------------------------------------------------- #
# cross-source de-duplication
# --------------------------------------------------------------------------- #
# Preferred source per shared paper (lower = kept as the cluster's primary row).
# Europe PMC first (full text + MeSH), then OpenAlex, arXiv, patents.
_SOURCE_RANK = "CASE source WHEN 'europepmc' THEN 0 WHEN 'openalex' THEN 1 " \
               "WHEN 'arxiv' THEN 2 WHEN 'patents' THEN 3 WHEN 'surechembl' THEN 4 " \
               "WHEN 'epo_ops' THEN 5 WHEN 'google_patents' THEN 6 ELSE 9 END"


def build_clusters(con: duckdb.DuckDBPyConnection | None = None) -> dict[str, int]:
    """Cluster `documents_raw` rows that share a DOI into one canonical paper.

    Builds `doc_clusters(pmcid, source, doi_norm, cluster_id, is_primary)`. Rows
    without a DOI cluster alone. `is_primary` marks the preferred row per cluster, so
    downstream analytics count unique papers (DISTINCT cluster_id) instead of
    source-rows. Returns {rows, clusters, duplicates}.
    """
    owns = con is None
    con = con or connect()
    try:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE doc_clusters AS
            WITH norm AS (
                SELECT pmcid, source, nullif(lower(trim(doi)), '') AS doi_norm
                FROM documents_raw
            ),
            ranked AS (
                SELECT pmcid, source, doi_norm,
                       coalesce(doi_norm, pmcid) AS cluster_id,
                       row_number() OVER (PARTITION BY coalesce(doi_norm, pmcid)
                                          ORDER BY {_SOURCE_RANK}, pmcid) AS rn
                FROM norm
            )
            SELECT pmcid, source, doi_norm, cluster_id, (rn = 1) AS is_primary
            FROM ranked
            """
        )
        rows = con.execute("SELECT count(*) FROM doc_clusters").fetchone()[0]
        clusters = con.execute("SELECT count(DISTINCT cluster_id) FROM doc_clusters").fetchone()[0]
        dupes = rows - clusters
        print(f"[dedup]   {rows} source-rows -> {clusters} unique papers ({dupes} cross-source dupes)")
        return {"rows": rows, "clusters": clusters, "duplicates": dupes}
    finally:
        if owns:
            con.close()


# --------------------------------------------------------------------------- #
# silver
# --------------------------------------------------------------------------- #
def process_documents(con: duckdb.DuckDBPyConnection | None = None) -> int:
    """Parse raw XML into `doc_sections`. Returns the section count."""
    owns = con is None
    con = con or connect()
    try:
        con.execute(
            """
            CREATE OR REPLACE TABLE doc_sections (
                pmcid TEXT, ordinal INTEGER, sec_type TEXT, sec_kind TEXT,
                sec_title TEXT, text TEXT, n_words INTEGER,
                n_figures INTEGER, n_tables INTEGER
            )
            """
        )
        docs = con.execute("SELECT pmcid, source, raw_xml FROM documents_raw").fetchall()
        rows = []
        for pmcid, source, xml in docs:
            parse = PARSERS.get(source, jats.parse_jats)
            parsed = parse(xml)
            for ordinal, sec in enumerate(parsed["sections"]):
                kind = section_kind(sec["sec_type"], sec.get("sec_title"))
                rows.append(
                    [pmcid, ordinal, sec["sec_type"], kind, sec["sec_title"],
                     sec["text"], len(sec["text"].split()),
                     int(sec.get("n_figures", 0)), int(sec.get("n_tables", 0))])
        if rows:
            con.executemany("INSERT INTO doc_sections VALUES (?,?,?,?,?,?,?,?,?)", rows)
        print(f"[process] {len(docs)} docs -> {len(rows)} sections (silver)")
        return len(rows)
    finally:
        if owns:
            con.close()


# --------------------------------------------------------------------------- #
# gold
# --------------------------------------------------------------------------- #
def _chunk(words: list[str], size: int, overlap: int):
    """Fixed word-window chunks with overlap (no boundary awareness)."""
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        window = words[i : i + size]
        if window:
            yield window
        if i + size >= len(words):
            break


def _chunk_sentences(text: str, size: int, overlap: int):
    """Sentence-boundary-aware chunks: pack whole sentences up to `size` words,
    carry ~`overlap` trailing words (whole sentences) into the next chunk.

    Falls back to fixed word-windows for any single sentence longer than `size`
    (e.g. unpunctuated PDF-extracted text), so chunks never grow unbounded.
    """
    sents = [s for s in _SENT.split(text) if s.strip()]
    if not sents:
        return
    cur: list[str] = []          # words in the current window
    carry: list[list[str]] = []  # trailing sentences to seed the next window
    for sent in sents:
        sw = sent.split()
        if len(sw) > size:  # monster sentence: emit current, then hard-split it
            if cur:
                yield cur
                cur = []
            yield from _chunk(sw, size, overlap)
            carry = []
            continue
        if cur and len(cur) + len(sw) > size:
            yield cur
            # rebuild the overlap from whole trailing sentences
            cur = [w for s in carry for w in s]
            carry = []
        cur.extend(sw)
        carry.append(sw)
        while sum(len(s) for s in carry) > overlap and len(carry) > 1:
            carry.pop(0)
    if cur:
        yield cur


def chunk_documents(con: duckdb.DuckDBPyConnection | None = None, *,
                    size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP,
                    sentence_aware: bool = CHUNK_SENTENCE_AWARE) -> int:
    """Split abstract + body sections into `doc_chunks`. Returns chunk count.

    Each chunk carries its section's structural metadata (kind, methods/results
    flags, figure/table counts) so retrieval can filter by document structure.
    Chunking is sentence-boundary-aware by default and tunable via `size`/`overlap`.
    """
    owns = con is None
    con = con or connect()
    try:
        con.execute(
            """
            CREATE OR REPLACE TABLE doc_chunks (
                pmcid TEXT, chunk_id INTEGER, sec_type TEXT, sec_kind TEXT,
                sec_title TEXT, text TEXT, n_words INTEGER,
                is_methods BOOLEAN, is_results BOOLEAN,
                n_figures INTEGER, n_tables INTEGER
            )
            """
        )
        sections = con.execute(
            """
            SELECT pmcid, sec_type, sec_kind, sec_title, text, n_figures, n_tables
            FROM doc_sections
            WHERE sec_type IN ('abstract', 'body')
            ORDER BY pmcid, ordinal
            """
        ).fetchall()

        splitter = _chunk_sentences if sentence_aware else (
            lambda t, s, o: _chunk(t.split(), s, o))
        counters: dict[str, int] = {}
        rows = []
        for pmcid, sec_type, sec_kind, sec_title, text, n_fig, n_tab in sections:
            for window in splitter(text, size, overlap):
                cid = counters.get(pmcid, 0)
                counters[pmcid] = cid + 1
                rows.append(
                    [pmcid, cid, sec_type, sec_kind, sec_title, " ".join(window),
                     len(window), sec_kind == "methods", sec_kind == "results",
                     int(n_fig or 0), int(n_tab or 0)])
        if rows:  # one batched insert beats tens of thousands of single-row appends
            con.executemany(
                "INSERT INTO doc_chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        print(f"[chunk]   {len(rows)} chunks -> doc_chunks (gold)")
        return len(rows)
    finally:
        if owns:
            con.close()


# --------------------------------------------------------------------------- #
# analytics + search
# --------------------------------------------------------------------------- #
def report(con: duckdb.DuckDBPyConnection | None = None) -> None:
    owns = con is None
    con = con or connect()
    try:
        d = con.execute(
            "SELECT count(*), count(*) FILTER (WHERE has_body), "
            "count(DISTINCT journal) FROM documents_raw"
        ).fetchone()
        s = con.execute("SELECT count(*), sum(n_words) FROM doc_sections").fetchone()
        c = con.execute("SELECT count(*) FROM doc_chunks").fetchone()[0]
        print("\n========== Aqueduct corpus ==========\n")
        print(f"Documents: {d[0]}  (full-text: {d[1]})   Journals: {d[2]}")
        print(f"Sections:  {s[0]}   Words: {s[1] or 0:,}   Chunks: {c}\n")
        print("-- Documents by year --")
        for yr, n, _w in con.execute(
            "SELECT pub_year, count(*), sum(LENGTH(raw_xml)) FROM documents_raw "
            "GROUP BY pub_year ORDER BY pub_year DESC"
        ).fetchall():
            print(f"  {yr}: {n} docs")
        print("\n-- Top journals --")
        for jr, n in con.execute(
            "SELECT journal, count(*) FROM documents_raw GROUP BY journal "
            "ORDER BY count(*) DESC LIMIT 5"
        ).fetchall():
            print(f"  {n:3}  {jr}")
        # MeSH terms (Europe PMC) — exploded from the ';'-joined column
        mesh = con.execute(
            """
            SELECT trim(term) AS t, count(*) AS n
            FROM documents_raw, UNNEST(string_split(mesh, ';')) AS u(term)
            WHERE mesh IS NOT NULL AND trim(term) <> ''
            GROUP BY t ORDER BY n DESC LIMIT 8
            """
        ).fetchall()
        if mesh:
            print("\n-- Top MeSH terms --")
            for term, n in mesh:
                print(f"  {n:3}  {term}")
        print("\n=====================================\n")
    finally:
        if owns:
            con.close()


def search(term: str, k: int = 8, con: duckdb.DuckDBPyConnection | None = None) -> None:
    """Lexical full-text search over chunks (placeholder for vector search)."""
    owns = con is None
    con = con or connect()
    try:
        rows = con.execute(
            """
            SELECT d.pmcid, d.title, c.sec_title, c.text
            FROM doc_chunks c JOIN documents_raw d USING (pmcid)
            WHERE c.text ILIKE '%' || ? || '%'
            LIMIT ?
            """,
            [term, k],
        ).fetchall()
        print(f"\n{len(rows)} chunk hits for {term!r}:\n")
        for pmcid, title, sec_title, text in rows:
            lo = text.lower().find(term.lower())
            snip = text[max(0, lo - 60) : lo + 120].strip()
            print(f"• {pmcid} — {(title or '')[:55]}")
            print(f"    [{sec_title or 'body'}] …{snip}…\n")
    finally:
        if owns:
            con.close()
