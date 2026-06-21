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


def _chunks(con, query: str, k: int, *, min_score: float = 0.0,
            sources: list[str] | None = None, sec_types: list[str] | None = None,
            kinds: list[str] | None = None) -> list[dict]:
    # over-fetch candidates, then apply metadata/score filters, then take top-k
    out = []
    for pmcid, cid, score in embeddings.rank(query, k=k * 4):
        if score < min_score:
            continue
        row = con.execute(
            """
            SELECT d.title, d.doi, d.source, d.pub_year, c.sec_type, c.sec_kind,
                   c.sec_title, c.is_methods, c.is_results, c.n_figures, c.n_tables, c.text
            FROM doc_chunks c JOIN documents_raw d USING (pmcid)
            WHERE c.pmcid = ? AND c.chunk_id = ?
            """, [pmcid, cid]).fetchone()
        if not row:
            continue
        title, doi, source, year, sec_type, sec_kind, sec, is_m, is_r, n_fig, n_tab, text = row
        if sources and source not in sources:
            continue
        if sec_types and sec_type not in sec_types:
            continue
        if kinds and sec_kind not in kinds:
            continue
        out.append({
            "id": pmcid, "title": title, "doi": doi, "source": source,
            "year": year, "sec_type": sec_type, "sec_kind": sec_kind, "section": sec,
            "is_methods": bool(is_m), "is_results": bool(is_r),
            "n_figures": int(n_fig or 0), "n_tables": int(n_tab or 0),
            "score": round(score, 4), "text": text,
        })
        if len(out) >= k:
            break
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


def retrieve(query: str, k: int = 8, graph: bool = True, con=None, *,
             min_score: float = 0.0, sources: list[str] | None = None,
             sec_types: list[str] | None = None, kinds: list[str] | None = None) -> dict:
    """Return grounded RAG context for a query: {query, n, chunks[], graph}.

    Optional filters (for callers like the Pardalos agent): `min_score` (drop weak
    matches), `sources` (e.g. ['europepmc']), `sec_types` (e.g. ['abstract']),
    `kinds` (IMRaD structure, e.g. ['methods', 'results']). Each returned chunk now
    carries `sec_kind`, `is_methods`/`is_results`, and `n_figures`/`n_tables`.
    """
    owns = con is None
    con = con or connect()
    try:
        chunks = _chunks(con, query, k, min_score=min_score, sources=sources,
                         sec_types=sec_types, kinds=kinds)
        info = embeddings.index_info()
        result = {"query": query, "n": len(chunks), "chunks": chunks,
                  "index": {"backend": info.get("backend"), "n_chunks": info.get("n_chunks")}}
        if graph:
            result["graph"] = _graph_context(con, query)
        if not chunks:
            result["note"] = "no semantic index or no matches — try `corpus index` or a different query"
        return result
    finally:
        if owns:
            con.close()
