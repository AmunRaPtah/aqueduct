"""RAG retrieval surface — structured (JSON) output for programmatic consumers.

This is the integration point for other systems (e.g. the Pardalos agent): a single
call returns the top semantically-relevant chunks with real citations, plus light
graph context when the query names a known drug or gene. Federated RAG — Aqueduct
retrieves with its OWN embeddings, so callers needn't share a vector space; they just
send a query string and get back grounded, citeable context.
"""

from __future__ import annotations

from . import embeddings, links
from .storage import connect


def _chunks(con, query: str, k: int) -> list[dict]:
    out = []
    for pmcid, cid, score in embeddings.rank(query, k=k):
        row = con.execute(
            """
            SELECT d.title, d.doi, d.source, d.pub_year, c.sec_title, c.text
            FROM doc_chunks c JOIN documents_raw d USING (pmcid)
            WHERE c.pmcid = ? AND c.chunk_id = ?
            """, [pmcid, cid]).fetchone()
        if not row:
            continue
        title, doi, source, year, sec, text = row
        out.append({
            "id": pmcid, "title": title, "doi": doi, "source": source,
            "year": year, "section": sec, "score": round(score, 4),
            "text": text,
        })
    return out


def _graph_context(con, query: str) -> dict:
    """If the query names a known drug or gene, attach its graph neighbourhood."""
    have = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    ctx: dict = {}
    ql = query.lower()
    if "entity_drug_names" in have:
        for (term,) in con.execute("SELECT DISTINCT term FROM entity_drug_names").fetchall():
            if term and len(term) >= 5 and term in ql:
                row = con.execute(
                    "SELECT drug_norm FROM entity_drug_names WHERE term=? LIMIT 1", [term]).fetchone()
                drug = row[0]
                targets = [r[0] for r in con.execute(
                    "SELECT DISTINCT gene FROM link_drug_protein WHERE drug_norm=?", [drug]).fetchall()] \
                    if "link_drug_protein" in have else []
                trials = con.execute(
                    "SELECT count(DISTINCT nct_id) FROM link_drug_trial WHERE drug_norm=? AND in_intervention",
                    [drug]).fetchone()[0] if "link_drug_trial" in have else 0
                ctx.setdefault("drugs", []).append(
                    {"drug": drug, "matched": term, "targets": targets, "trials": trials})
                break
    if "entity_proteins" in have:
        for (gene,) in con.execute(
                "SELECT DISTINCT gene FROM entity_proteins WHERE gene IS NOT NULL").fetchall():
            if gene and len(gene) >= 3 and gene.lower() in ql:
                drugs = [r[0] for r in con.execute(
                    "SELECT DISTINCT drug_norm FROM link_drug_protein WHERE lower(gene)=?",
                    [gene.lower()]).fetchall()] if "link_drug_protein" in have else []
                ctx.setdefault("genes", []).append({"gene": gene, "drugs": drugs})
                break
    return ctx


def retrieve(query: str, k: int = 8, graph: bool = True, con=None) -> dict:
    """Return grounded RAG context for a query: {query, n, chunks[], graph}."""
    owns = con is None
    con = con or connect()
    try:
        chunks = _chunks(con, query, k)
        result = {"query": query, "n": len(chunks), "chunks": chunks}
        if graph:
            result["graph"] = _graph_context(con, query)
        if not chunks:
            result["note"] = "no semantic index or no matches — try `corpus index` or a different query"
        return result
    finally:
        if owns:
            con.close()
