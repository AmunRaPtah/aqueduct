"""Data-quality validation over the corpus (a non-fatal QA pass).

Harvesting from many noisy sources can introduce malformed rows — empty titles,
body-less "full-text" docs, zero-word sections, oversized chunks, bad ids. This
pass *reports* them (counts + a few examples) so they surface in logs instead of
silently degrading retrieval; it never raises on bad data and never mutates the
warehouse. Run it after a build (harvest does, automatically).
"""

from __future__ import annotations

import re

import duckdb

from . import obs
from .documents import CHUNK_WORDS
from .storage import connect

# Accepted document-id shapes per source (the `pmcid` column is the shared id slot).
_ID_PATTERNS = {
    "europepmc": re.compile(r"^PMC\d+$"),
    "arxiv": re.compile(r"^arXiv:.+", re.I),
    "openalex": re.compile(r"^(openalex:|https?://openalex\.org/)?W\d+$", re.I),
    "patents": re.compile(r".+"),  # patent numbers vary; just require non-empty
    "surechembl": re.compile(r".+"),
    "epo_ops": re.compile(r".+"),
    "google_patents": re.compile(r".+"),
}

_YEAR_MIN, _YEAR_MAX = 1500, 2100  # sanity bounds for pub_year


def validate(con: duckdb.DuckDBPyConnection | None = None, *, verbose: bool = True) -> dict:
    """Return a structured data-quality report: {checks{name: count}, examples, ok}."""
    owns = con is None
    con = con or connect()
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "documents_raw" not in tables:
            if verbose:
                print("[validate] no corpus yet — nothing to check")
            return {"checks": {}, "examples": {}, "ok": True, "n_documents": 0}

        checks: dict[str, int] = {}
        examples: dict[str, list] = {}

        def count(name: str, sql: str, ex_sql: str | None = None) -> None:
            checks[name] = con.execute(sql).fetchone()[0]
            if checks[name] and ex_sql:
                examples[name] = [r[0] for r in con.execute(ex_sql).fetchall()]

        count("missing_title",
              "SELECT count(*) FROM documents_raw WHERE title IS NULL OR trim(title)=''",
              "SELECT pmcid FROM documents_raw WHERE title IS NULL OR trim(title)='' LIMIT 5")
        count("year_out_of_range",
              f"SELECT count(*) FROM documents_raw WHERE pub_year IS NOT NULL "
              f"AND (pub_year < {_YEAR_MIN} OR pub_year > {_YEAR_MAX})",
              f"SELECT pmcid FROM documents_raw WHERE pub_year IS NOT NULL "
              f"AND (pub_year < {_YEAR_MIN} OR pub_year > {_YEAR_MAX}) LIMIT 5")
        # has_body=True but the parser produced no body sections (a broken full-text fetch)
        if "doc_sections" in tables:
            count("body_flag_without_sections",
                  "SELECT count(*) FROM documents_raw d WHERE d.has_body "
                  "AND NOT EXISTS (SELECT 1 FROM doc_sections s "
                  "WHERE s.pmcid=d.pmcid AND s.sec_type='body')",
                  "SELECT d.pmcid FROM documents_raw d WHERE d.has_body "
                  "AND NOT EXISTS (SELECT 1 FROM doc_sections s "
                  "WHERE s.pmcid=d.pmcid AND s.sec_type='body') LIMIT 5")
            count("empty_sections",
                  "SELECT count(*) FROM doc_sections WHERE n_words = 0 OR text IS NULL OR trim(text)=''")
        if "doc_chunks" in tables:
            count("oversized_chunks",
                  f"SELECT count(*) FROM doc_chunks WHERE n_words > {2 * CHUNK_WORDS}")
            count("empty_chunks", "SELECT count(*) FROM doc_chunks WHERE n_words = 0")

        # id-format check (Python regex per source, since patterns differ)
        bad_ids = []
        for pmcid, source in con.execute(
                "SELECT pmcid, source FROM documents_raw").fetchall():
            pat = _ID_PATTERNS.get(source)
            if not pmcid or (pat and not pat.match(str(pmcid))):
                bad_ids.append(pmcid)
        checks["bad_id_format"] = len(bad_ids)
        if bad_ids:
            examples["bad_id_format"] = bad_ids[:5]

        n_docs = con.execute("SELECT count(*) FROM documents_raw").fetchone()[0]
        n_issues = sum(checks.values())
        report = {"checks": checks, "examples": examples, "ok": n_issues == 0,
                  "n_documents": n_docs, "n_issues": n_issues}
        obs.log("validate", n_documents=n_docs, n_issues=n_issues, checks=checks)
        if verbose:
            flagged = {k: v for k, v in checks.items() if v}
            if not flagged:
                print(f"[validate] {n_docs} docs — no issues found")
            else:
                print(f"[validate] {n_docs} docs — {n_issues} issue(s) across "
                      f"{len(flagged)} check(s):")
                for name, n in sorted(flagged.items(), key=lambda kv: -kv[1]):
                    ex = examples.get(name, [])
                    tail = f"  e.g. {', '.join(map(str, ex))}" if ex else ""
                    print(f"  {n:5}  {name}{tail}")
        return report
    finally:
        if owns:
            con.close()
