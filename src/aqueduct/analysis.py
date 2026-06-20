"""Quantitative analysis over the warehouse — DuckDB does the math (zero LLM tokens).

Every function returns plain JSON-able structures. `facts()` aggregates them into one
dict and `facts_sheet()` renders a compact markdown brief — that brief is the cheap fuel
fed to the LLM / agentic analyst, so it never spends tokens rediscovering basic numbers.
All queries are guarded by table presence, so partial corpora still work.
"""

from __future__ import annotations

import duckdb

from .storage import connect


def _has(con, t: str) -> bool:
    return t in {r[0] for r in con.execute("SHOW TABLES").fetchall()}


def _rows(con, sql: str, params=None) -> list[list]:
    return [list(r) for r in con.execute(sql, params or []).fetchall()]


# --------------------------------------------------------------------------- #
def overview(con) -> dict:
    o: dict = {}
    if _has(con, "doc_clusters"):
        o["unique_papers"] = con.execute(
            "SELECT count(DISTINCT cluster_id) FROM doc_clusters").fetchone()[0]
        o["source_rows"] = con.execute("SELECT count(*) FROM doc_clusters").fetchone()[0]
        o["cross_source_dupes"] = o["source_rows"] - o["unique_papers"]
    if _has(con, "documents_raw"):
        o["by_source"] = _rows(con,
            "SELECT source, count(*) n FROM documents_raw GROUP BY source ORDER BY n DESC")
        o["full_text"] = con.execute(
            "SELECT count(*) FROM documents_raw WHERE has_body").fetchone()[0]
    if _has(con, "doc_chunks"):
        o["chunks"] = con.execute("SELECT count(*) FROM doc_chunks").fetchone()[0]
    return o


def trends(con) -> list[list]:
    """Unique papers per year (deduped via clusters' primary rows)."""
    if not (_has(con, "doc_clusters") and _has(con, "documents_raw")):
        return []
    return _rows(con,
        """
        SELECT d.pub_year, count(DISTINCT c.cluster_id) papers
        FROM doc_clusters c JOIN documents_raw d USING (pmcid)
        WHERE c.is_primary AND d.pub_year IS NOT NULL
        GROUP BY d.pub_year ORDER BY d.pub_year
        """)


def top_drugs(con, k: int = 12) -> list[list]:
    if not _has(con, "entity_drugs"):
        return []
    return _rows(con,
        f"""
        SELECT e.drug_norm, e.max_phase,
               count(DISTINCT dt.nct_id) FILTER (WHERE dt.in_intervention) trials,
               count(DISTINCT dd.pmcid) FILTER (WHERE dd.confidence='strong') papers,
               count(DISTINCT dp.gene) targets
        FROM entity_drugs e
        LEFT JOIN link_drug_trial dt    USING (drug_norm)
        LEFT JOIN link_drug_document dd USING (drug_norm)
        LEFT JOIN link_drug_protein dp  USING (drug_norm)
        GROUP BY e.drug_norm, e.max_phase
        ORDER BY (trials + papers + targets) DESC LIMIT {k}
        """)


def top_targets(con, k: int = 12) -> list[list]:
    if not _has(con, "entity_proteins"):
        return []
    return _rows(con,
        f"""
        SELECT p.gene,
               count(DISTINCT dp.drug_norm) drugs,
               count(DISTINCT ps.pdb_id) structures,
               count(DISTINCT pd.pmcid) FILTER (WHERE pd.confidence='strong') papers
        FROM entity_proteins p
        LEFT JOIN link_drug_protein dp     ON dp.gene = p.gene
        LEFT JOIN link_protein_structure ps ON ps.gene = p.gene
        LEFT JOIN link_protein_document pd  ON pd.gene = p.gene
        WHERE p.gene IS NOT NULL
        GROUP BY p.gene ORDER BY (drugs + structures + papers) DESC LIMIT {k}
        """)


def gaps(con, k: int = 8) -> dict:
    """Whitespace: entities missing an expected connection (research opportunities)."""
    g: dict = {}
    if _has(con, "entity_proteins") and _has(con, "link_drug_protein"):
        g["targets_without_drugs"] = _rows(con,
            f"""SELECT gene FROM entity_proteins
                WHERE gene IS NOT NULL AND gene NOT IN
                  (SELECT gene FROM link_drug_protein WHERE gene IS NOT NULL)
                LIMIT {k}""")
    if _has(con, "entity_drugs") and _has(con, "link_drug_trial"):
        g["drugs_without_trials"] = _rows(con,
            f"""SELECT drug_norm, max_phase FROM entity_drugs
                WHERE drug_norm NOT IN
                  (SELECT drug_norm FROM link_drug_trial WHERE in_intervention)
                ORDER BY max_phase DESC NULLS LAST LIMIT {k}""")
    if _has(con, "entity_drugs") and _has(con, "link_drug_document"):
        g["drugs_without_papers"] = _rows(con,
            f"""SELECT drug_norm FROM entity_drugs
                WHERE drug_norm NOT IN
                  (SELECT drug_norm FROM link_drug_document WHERE confidence='strong')
                LIMIT {k}""")
    return g


def asymmetries(con, k: int = 8) -> dict:
    """Mismatches in the data — the raw material for novel insight, not summaries.

    Structurally-studied-but-undrugged targets, research-rich-but-untrialed drugs,
    and drugs spanning many conditions (repurposing signals).
    """
    a: dict = {}
    if _has(con, "entity_proteins") and _has(con, "link_protein_structure") and _has(con, "link_drug_protein"):
        a["structurally_studied_but_undrugged"] = _rows(con,
            f"""SELECT p.gene, count(DISTINCT ps.pdb_id) structures,
                       count(DISTINCT pd.pmcid) FILTER (WHERE pd.confidence='strong') papers
                FROM entity_proteins p
                JOIN link_protein_structure ps ON ps.gene = p.gene
                LEFT JOIN link_protein_document pd ON pd.gene = p.gene
                WHERE p.gene NOT IN (SELECT gene FROM link_drug_protein WHERE gene IS NOT NULL)
                GROUP BY p.gene HAVING count(DISTINCT ps.pdb_id) >= 3
                ORDER BY structures DESC LIMIT {k}""")
    if _has(con, "entity_drugs") and _has(con, "link_drug_document") and _has(con, "link_drug_trial"):
        a["research_rich_but_untrialed"] = _rows(con,
            f"""SELECT e.drug_norm, count(DISTINCT dd.pmcid) papers, e.max_phase
                FROM entity_drugs e JOIN link_drug_document dd USING (drug_norm)
                WHERE dd.confidence='strong' AND e.drug_norm NOT IN
                  (SELECT drug_norm FROM link_drug_trial WHERE in_intervention)
                GROUP BY e.drug_norm, e.max_phase HAVING count(DISTINCT dd.pmcid) >= 3
                ORDER BY papers DESC LIMIT {k}""")
    if _has(con, "link_drug_trial") and _has(con, "clinical_trials"):
        a["drugs_spanning_many_conditions"] = _rows(con,
            f"""SELECT l.drug_norm, count(DISTINCT t.conditions) conditions
                FROM link_drug_trial l JOIN clinical_trials t USING (nct_id)
                WHERE t.conditions IS NOT NULL AND l.in_intervention
                GROUP BY l.drug_norm HAVING count(DISTINCT t.conditions) >= 2
                ORDER BY conditions DESC LIMIT {k}""")
    return a


def trials(con) -> dict:
    if not _has(con, "clinical_trials"):
        return {}
    return {
        "by_phase": _rows(con,
            "SELECT phases, count(*) n FROM clinical_trials GROUP BY phases ORDER BY n DESC LIMIT 8"),
        "by_status": _rows(con,
            "SELECT status, count(*) n FROM clinical_trials GROUP BY status ORDER BY n DESC LIMIT 8"),
    }


def binding(con, k: int = 10) -> list[list]:
    if not _has(con, "binding_affinities"):
        return []
    return _rows(con,
        f"""SELECT target, affinity_type, count(*) n, round(min(affinity_nm), 2) best_nm
            FROM binding_affinities WHERE affinity_nm IS NOT NULL
            GROUP BY target, affinity_type ORDER BY best_nm LIMIT {k}""")


def topics(con, k: int = 12) -> dict:
    t: dict = {}
    if _has(con, "documents_raw"):
        t["mesh"] = _rows(con,
            f"""SELECT trim(term) t, count(*) n
                FROM documents_raw, UNNEST(string_split(mesh, ';')) u(term)
                WHERE mesh IS NOT NULL AND trim(term) <> ''
                GROUP BY t ORDER BY n DESC LIMIT {k}""")
        t["concepts"] = _rows(con,
            f"""SELECT trim(term) t, count(*) n
                FROM documents_raw, UNNEST(string_split(keywords, ';')) u(term)
                WHERE source='openalex' AND keywords IS NOT NULL AND trim(term) <> ''
                GROUP BY t ORDER BY n DESC LIMIT {k}""")
    return t


# --------------------------------------------------------------------------- #
def facts(con=None) -> dict:
    """Aggregate every metric into one structured facts object."""
    owns = con is None
    con = con or connect()
    try:
        return {
            "overview": overview(con),
            "trends": trends(con),
            "top_drugs": top_drugs(con),
            "top_targets": top_targets(con),
            "gaps": gaps(con),
            "asymmetries": asymmetries(con),
            "trials": trials(con),
            "binding": binding(con),
            "topics": topics(con),
        }
    finally:
        if owns:
            con.close()


def _tbl(rows: list[list], headers: list[str]) -> str:
    if not rows:
        return "_none_"
    out = [" | ".join(headers), " | ".join("---" for _ in headers)]
    out += [" | ".join(str(c) for c in r) for r in rows]
    return "\n".join(out)


def facts_sheet(con=None) -> str:
    """Render the facts as a compact markdown brief (LLM input + report data section)."""
    f = facts(con)
    o = f["overview"]
    s = ["# Aqueduct facts\n"]
    s.append(f"- Unique papers: {o.get('unique_papers','?')} "
             f"(from {o.get('source_rows','?')} source-rows, "
             f"{o.get('cross_source_dupes','?')} cross-source dupes); "
             f"full-text: {o.get('full_text','?')}; chunks: {o.get('chunks','?')}")
    if o.get("by_source"):
        s.append("- By source: " + ", ".join(f"{src}={n}" for src, n in o["by_source"]))
    s.append("\n## Papers per year\n" + _tbl(f["trends"], ["year", "papers"]))
    s.append("\n## Top drugs (by connectivity)\n"
             + _tbl(f["top_drugs"], ["drug", "max_phase", "trials", "papers", "targets"]))
    s.append("\n## Top targets\n"
             + _tbl(f["top_targets"], ["gene", "drugs", "structures", "papers"]))
    g = f["gaps"]
    s.append("\n## Gaps")
    s.append("- Targets without drugs: "
             + ", ".join(r[0] for r in g.get("targets_without_drugs", [])) or "_none_")
    s.append("- Drugs without trials: "
             + ", ".join(r[0] for r in g.get("drugs_without_trials", [])) or "_none_")
    a = f.get("asymmetries", {})
    if a:
        s.append("\n## Asymmetries (novelty signals)")
        s.append("- Structurally studied but UNDRUGGED targets (gene, #structures, #papers): "
                 + "; ".join(f"{r[0]}({r[1]}str,{r[2]}pap)"
                             for r in a.get("structurally_studied_but_undrugged", [])) or "_none_")
        s.append("- Research-rich but UNTRIALED drugs (drug, #papers): "
                 + "; ".join(f"{r[0]}({r[1]})" for r in a.get("research_rich_but_untrialed", [])) or "_none_")
        s.append("- Drugs spanning many conditions (repurposing) (drug, #conditions): "
                 + "; ".join(f"{r[0]}({r[1]})" for r in a.get("drugs_spanning_many_conditions", [])) or "_none_")
    if f["trials"]:
        s.append("\n## Clinical trials by phase\n"
                 + _tbl(f["trials"].get("by_phase", []), ["phase", "n"]))
    if f["binding"]:
        s.append("\n## Strongest binding (lowest nM)\n"
                 + _tbl(f["binding"], ["target", "type", "n", "best_nM"]))
    t = f["topics"]
    if t.get("mesh"):
        s.append("\n## Top MeSH terms: " + ", ".join(f"{x[0]}({x[1]})" for x in t["mesh"]))
    if t.get("concepts"):
        s.append("## Top OpenAlex concepts: " + ", ".join(f"{x[0]}({x[1]})" for x in t["concepts"]))
    return "\n".join(s)
