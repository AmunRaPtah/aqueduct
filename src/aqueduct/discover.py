"""Semantic-over-graph discovery.

Uses the knowledge graph to assemble a *concept profile* for a drug — its target
protein names + function, mechanism of action, and trial conditions — then runs that
profile through semantic search over the corpus. The payoff: papers that are
conceptually about the drug's biology surface even when they never name the drug, and
each hit is tagged DIRECT (already a lexical link) or SEMANTIC (newly discovered).

This is the join of the two big subsystems: the entity-resolved link graph and the
chunk embeddings.
"""

from __future__ import annotations

from . import embeddings, links
from .storage import connect


def _profile(con, drug_norm: str) -> dict:
    """Gather graph context for a drug into a concept profile (text + facts).

    Sources may be absent (a corpus need not include every dataset), so each part is
    guarded by whether its table was loaded.
    """
    have = {r[0] for r in con.execute("SHOW TABLES").fetchall()}

    names = [r[0] for r in con.execute(
        "SELECT term FROM entity_drug_names WHERE drug_norm=?", [drug_norm]).fetchall()] \
        if "entity_drug_names" in have else []
    mechanisms = [r[0] for r in con.execute(
        """
        SELECT DISTINCT m.mechanism_of_action
        FROM entity_drug_molecules e JOIN chembl_mechanisms m
          ON e.chembl_id = m.molecule_chembl_id
        WHERE e.drug_norm = ? AND m.mechanism_of_action IS NOT NULL
        """, [drug_norm]).fetchall()] if "chembl_mechanisms" in have else []
    proteins = con.execute(
        """
        SELECT DISTINCT u.protein_name, u.function, u.aliases
        FROM link_drug_protein l JOIN uniprot_proteins u USING (accession)
        WHERE l.drug_norm = ?
        """, [drug_norm]).fetchall() if "uniprot_proteins" in have else []
    conditions = [r[0] for r in con.execute(
        """
        SELECT DISTINCT t.conditions FROM link_drug_trial l
        JOIN clinical_trials t USING (nct_id)
        WHERE l.drug_norm = ? AND t.conditions IS NOT NULL
        """, [drug_norm]).fetchall()] if "clinical_trials" in have else []

    parts = list(names) + list(mechanisms) + list(conditions)
    for pname, func, aliases in proteins:
        parts += [p for p in (pname, func, aliases) if p]
    return {
        "text": ". ".join(p for p in parts if p),
        "mechanisms": mechanisms,
        "proteins": [p[0] for p in proteins],
        "conditions": conditions,
    }


def ranked_papers(name: str, k: int = 8, con=None):
    """Return (profile, [(pmcid, score, is_direct), ...]) for a drug, ranked semantically."""
    owns = con is None
    con = con or connect()
    try:
        norm = links._norm(name) or name.lower()
        prof = _profile(con, norm)
        if not prof["text"]:
            return prof, []
        hits = embeddings.rank(prof["text"], k=k * 8)
        best: dict[str, float] = {}
        for pmcid, _cid, score in hits:
            if score > best.get(pmcid, -1):
                best[pmcid] = score
        direct = {r[0] for r in con.execute(
            "SELECT DISTINCT pmcid FROM link_drug_document WHERE drug_norm=? AND confidence='strong'",
            [norm]).fetchall()}
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return prof, [(pmcid, score, pmcid in direct) for pmcid, score in ranked]
    finally:
        if owns:
            con.close()


def drug(name: str, k: int = 8, con=None) -> None:
    """Discover literature conceptually about a drug's biology via the graph + embeddings."""
    owns = con is None
    con = con or connect()
    try:
        norm = links._norm(name) or name.lower()
        prof, ranked = ranked_papers(name, k=k, con=con)
        if not prof["text"]:
            print(f"No graph context for {name!r} (build the graph: `links build`).")
            return
        print(f"\n=== Discover: {name}  (canonical '{norm}') ===")
        if prof["proteins"]:
            print(f"  targets:    {', '.join(prof['proteins'])}")
        if prof["mechanisms"]:
            print(f"  mechanism:  {'; '.join(prof['mechanisms'])[:80]}")
        if prof["conditions"]:
            print(f"  conditions: {'; '.join(prof['conditions'])[:80]}")
        if not ranked:
            print("\n  (no semantic index — run `corpus index`)")
            return
        print(f"\nPapers conceptually about its biology ({len(ranked)}):")
        for pmcid, score, is_direct in ranked:
            title = con.execute(
                "SELECT title FROM documents_raw WHERE pmcid=?", [pmcid]).fetchone()
            tag = "DIRECT  " if is_direct else "SEMANTIC"
            print(f"  [{score:.3f}] {tag} {pmcid}  {(title[0] if title else '')[:50]}")
        n_new = sum(1 for _, _, d in ranked if not d)
        print(f"\n  {n_new}/{len(ranked)} surfaced by meaning beyond the lexical links.\n")
    finally:
        if owns:
            con.close()
