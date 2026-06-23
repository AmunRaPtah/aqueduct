"""Cross-source entity links with entity resolution (the "link later" step).

Connects the silos — `chembl_molecules`/`chembl_synonyms` (drugs), `clinical_trials`
(trials), `documents_raw`/`doc_sections` (literature) — into a graph, drug-as-hub.

Entity resolution:
- canonical drug = normalised base name (salt forms / hydrates stripped)
- match terms = base name + whitelisted synonyms / trade names (so Narcan->naloxone)
- match confidence on drug<->document links: a hit in title/abstract/keywords/MeSH, or
  >=2 full-text mentions, is `strong`; a single body co-mention is `weak`.

  entity_drugs         canonical drug + form count + max clinical phase
  entity_drug_names    canonical drug -> all match terms (names + synonyms)
  link_drug_trial      drug <-> trial  (+ in_intervention flag)
  link_drug_document   drug <-> paper  (+ in_metadata, n_body, confidence)
  link_trial_document  trial <-> paper, via a shared drug (intervention + strong only)
"""

from __future__ import annotations

import re

import duckdb

from .storage import connect

_SALTS = {
    "hydrochloride", "hcl", "hydrobromide", "sulfate", "sulphate", "bitartrate",
    "tartrate", "citrate", "maleate", "mesylate", "besylate", "fumarate",
    "phosphate", "acetate", "succinate", "hydrate", "dihydrate", "monohydrate",
    "sodium", "potassium", "calcium", "bromide", "chloride", "nitrate",
    "polacrilex", "anhydrous", "base", "pamoate", "decanoate", "valerate",
}
# leading stereochemistry / isomer descriptors — strip so enantiomers and racemates
# collapse onto the same canonical drug (e.g. "(R)-methadone" -> "methadone").
_STEREO = {
    "r", "s", "rs", "sr", "rac", "racemic", "d", "l", "dl", "ld",
    "cis", "trans", "levo", "dextro", "es", "ar", "ent",
}
# generic words that are also drug/ingredient names — too noisy to match on
_STOP = {
    "water", "oxygen", "alcohol", "glucose", "saline", "placebo", "control",
    "nitrogen", "carbon", "starch", "sucrose", "lactose", "glycerol", "dextrose",
}
_WORD = "(^|[^a-z])"  # left/right word boundary for lowercase regexp


def _norm(name: str | None) -> str | None:
    """Normalise a name to a base drug token (salt forms stripped)."""
    if not name:
        return None
    tokens = re.findall(r"[a-z]+", name.lower())
    base = [t for t in tokens if t not in _SALTS and t not in _STEREO]
    head = (base or tokens or [""])[0]
    return head if len(head) >= 5 and head not in _STOP else None


def _term(name: str | None) -> str | None:
    """A clean single-token match term from a synonym/trade name."""
    if not name:
        return None
    toks = re.findall(r"[a-z]+", name.lower())
    if len(toks) != 1:  # skip multiword/hyphenated/coded synonyms
        return None
    t = toks[0]
    return t if len(t) >= 5 and t not in _STOP else None


def _pat(col: str) -> str:
    """SQL fragment: word-boundary regexp of term `t.term` against lower(`col`)."""
    return f"regexp_matches(lower({col}), '{_WORD}' || n.term || '([^a-z]|$)')"


# generic words too vague to identify a protein on their own
_PROT_STOP = {
    "receptor", "protein", "factor", "kinase", "enzyme", "subunit", "channel",
    "transporter", "alpha", "beta", "gamma", "delta", "type", "human", "chain",
    "domain", "family", "like", "putative", "isoform", "precursor", "binding",
}
_GB = "(^|[^a-z0-9])"   # left boundary that also excludes digits (gene symbols have them)
_GBR = "([^a-z0-9]|$)"  # right boundary

# Numeric (0..1) confidence for a text<->document link. A metadata hit (title /
# abstract / keywords / MeSH) is worth a strong base; body mentions add a
# saturating bonus (one incidental mention stays low, repeated mentions approach
# certainty) — so callers get a graded signal, not just strong/weak.
_LINK_SCORE = ("round(least(1.0, (CASE WHEN {meta} THEN 0.6 ELSE 0 END) "
               "+ (1 - exp(-coalesce({nbody}, 0) / 1.5)) * 0.7), 3)")


def _protein_terms(gene: str | None, aliases: list[str]):
    """Yield (term, regexp_pattern) for a protein's gene symbol + name aliases.

    Single-token names need length/specificity; multiword names match with flexible
    separators (so "Mu opioid receptor" matches "mu-opioid receptor").
    """
    gene_l = (gene or "").lower()
    out: dict[str, str] = {}
    for cand in ([gene] if gene else []) + list(aliases):
        toks = re.findall(r"[a-z0-9]+", (cand or "").lower())
        if not toks:
            continue
        if len(toks) == 1:
            t = toks[0]
            specific = (len(t) >= 4 and t not in _PROT_STOP) or (t == gene_l and len(t) >= 3)
            if specific:
                out[t] = f"{_GB}{t}{_GBR}"
        elif any(len(t) >= 4 and t not in _PROT_STOP for t in toks):
            term = " ".join(toks)
            out[term] = _GB + "[^a-z0-9]+".join(toks) + _GBR
    return out.items()


def build(con: duckdb.DuckDBPyConnection | None = None) -> dict[str, int]:
    """Resolve drug entities + build all link tables. Returns row counts."""
    owns = con is None
    con = con or connect()
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}

        # ---- entity resolution: canonical drugs + their match terms ----
        con.execute("CREATE OR REPLACE TABLE entity_drug_molecules "
                    "(chembl_id TEXT, drug_norm TEXT, pref_name TEXT, max_phase DOUBLE)")
        terms: dict[str, set[str]] = {}
        cid_norm: dict[str, str] = {}
        if "chembl_molecules" in tables:
            for cid, name, phase in con.execute(
                "SELECT chembl_id, pref_name, max_phase FROM chembl_molecules"
            ).fetchall():
                norm = _norm(name)
                if not norm:
                    continue
                cid_norm[cid] = norm
                con.execute("INSERT INTO entity_drug_molecules VALUES (?,?,?,?)",
                            [cid, norm, name, phase])
                terms.setdefault(norm, set()).add(norm)
        if "chembl_synonyms" in tables:
            for cid, syn in con.execute(
                "SELECT chembl_id, name FROM chembl_synonyms"
            ).fetchall():
                norm = cid_norm.get(cid)
                t = _term(syn)
                if norm and t:
                    terms[norm].add(t)

        con.execute(
            """
            CREATE OR REPLACE TABLE entity_drugs AS
            SELECT drug_norm, count(DISTINCT chembl_id) AS n_forms,
                   max(max_phase) AS max_phase, min(pref_name) AS sample_name
            FROM entity_drug_molecules GROUP BY drug_norm
            """
        )
        con.execute("CREATE OR REPLACE TABLE entity_drug_names (drug_norm TEXT, term TEXT)")
        for norm, tset in terms.items():
            for t in tset:
                con.execute("INSERT INTO entity_drug_names VALUES (?,?)", [norm, t])

        # ---- document text blobs: metadata (strong) vs full body (counted) ----
        have_docs = "doc_sections" in tables
        if have_docs:
            con.execute(
                """
                CREATE OR REPLACE TEMP TABLE _doc_meta AS
                SELECT pmcid, lower(coalesce(title,'') || ' ' || coalesce(abstract,'') || ' '
                       || coalesce(keywords,'') || ' ' || coalesce(mesh,'')) AS blob
                FROM documents_raw
                """
            )
            con.execute(
                """
                CREATE OR REPLACE TEMP TABLE _doc_body AS
                SELECT pmcid, lower(coalesce(string_agg(text, ' '), '')) AS blob
                FROM doc_sections GROUP BY pmcid
                """
            )

        # ---- drug <-> trial ----
        con.execute("CREATE OR REPLACE TABLE link_drug_trial "
                    "(drug_norm TEXT, nct_id TEXT, in_intervention BOOLEAN)")
        if "clinical_trials" in tables:
            con.execute(
                f"""
                INSERT INTO link_drug_trial
                SELECT n.drug_norm, t.nct_id, bool_or({_pat("t.interventions")})
                FROM entity_drug_names n JOIN clinical_trials t
                  ON {_pat("coalesce(t.interventions,'') || ' ' || coalesce(t.title,'')")}
                GROUP BY n.drug_norm, t.nct_id
                """
            )

        # ---- drug <-> document (with confidence) ----
        con.execute("CREATE OR REPLACE TABLE link_drug_document "
                    "(drug_norm TEXT, pmcid TEXT, in_metadata BOOLEAN, n_body INTEGER, "
                    "confidence TEXT, score DOUBLE)")
        if have_docs:
            con.execute(
                f"""
                INSERT INTO link_drug_document
                WITH meta AS (
                    SELECT DISTINCT n.drug_norm, m.pmcid
                    FROM entity_drug_names n JOIN _doc_meta m ON {_pat("m.blob")}
                ),
                body AS (
                    SELECT n.drug_norm, b.pmcid,
                           sum(len(regexp_extract_all(b.blob, '{_WORD}' || n.term || '([^a-z]|$)'))) AS n_body
                    FROM entity_drug_names n JOIN _doc_body b ON {_pat("b.blob")}
                    GROUP BY n.drug_norm, b.pmcid
                )
                SELECT coalesce(m.drug_norm, b.drug_norm),
                       coalesce(m.pmcid, b.pmcid),
                       m.pmcid IS NOT NULL,
                       coalesce(b.n_body, 0),
                       CASE WHEN m.pmcid IS NOT NULL OR coalesce(b.n_body, 0) >= 2
                            THEN 'strong' ELSE 'weak' END,
                       {_LINK_SCORE.format(meta="m.pmcid IS NOT NULL", nbody="b.n_body")}
                FROM meta m FULL OUTER JOIN body b
                  ON m.drug_norm = b.drug_norm AND m.pmcid = b.pmcid
                """
            )

        # ---- trial <-> document via shared drug (intervention + strong doc only) ----
        con.execute(
            """
            CREATE OR REPLACE TABLE link_trial_document AS
            SELECT DISTINCT dt.nct_id, dd.pmcid, dt.drug_norm
            FROM link_drug_trial dt JOIN link_drug_document dd USING (drug_norm)
            WHERE dt.in_intervention AND dd.confidence = 'strong'
            """
        )

        # ---- protein arm (UniProt targets) ----
        have_prot = "uniprot_proteins" in tables
        con.execute("CREATE OR REPLACE TABLE link_drug_protein "
                    "(drug_norm TEXT, chembl_id TEXT, accession TEXT, gene TEXT, action_type TEXT)")
        con.execute("CREATE OR REPLACE TABLE link_protein_structure (accession TEXT, gene TEXT, pdb_id TEXT)")
        con.execute("CREATE OR REPLACE TABLE link_protein_document "
                    "(accession TEXT, gene TEXT, pmcid TEXT, in_metadata BOOLEAN, n_body INTEGER, "
                    "confidence TEXT, score DOUBLE)")
        if have_prot:
            con.execute(
                """
                CREATE OR REPLACE TABLE entity_proteins AS
                SELECT accession, gene, protein_name, organism, chembl_target,
                       lower(gene) AS gene_norm
                FROM uniprot_proteins WHERE accession IS NOT NULL
                """
            )
            # match-term dictionary: gene symbol + name aliases, each with its regexp
            con.execute("CREATE OR REPLACE TABLE entity_protein_names "
                        "(accession TEXT, gene TEXT, term TEXT, pattern TEXT)")
            for acc, gene, aliases in con.execute(
                "SELECT accession, gene, aliases FROM uniprot_proteins WHERE accession IS NOT NULL"
            ).fetchall():
                alist = [a.strip() for a in (aliases or "").split(";") if a.strip()]
                for term, pat in _protein_terms(gene, alist):
                    con.execute("INSERT INTO entity_protein_names VALUES (?,?,?,?)",
                                [acc, gene, term, pat])
            # drug -> target protein, via ChEMBL mechanism <-> UniProt ChEMBL-target xref
            if "chembl_mechanisms" in tables:
                con.execute(
                    """
                    INSERT INTO link_drug_protein
                    SELECT DISTINCT mn.drug_norm, x.molecule_chembl_id, u.accession, u.gene, x.action_type
                    FROM chembl_mechanisms x
                    JOIN uniprot_proteins u ON x.target_chembl_id = u.chembl_target
                    JOIN entity_drug_molecules mn ON mn.chembl_id = x.molecule_chembl_id
                    """
                )
            # protein -> structure, from UniProt PDB cross-references
            con.execute(
                """
                INSERT INTO link_protein_structure
                SELECT DISTINCT accession, gene, trim(pid)
                FROM uniprot_proteins, UNNEST(string_split(coalesce(pdb_ids,''), ';')) AS u(pid)
                WHERE trim(pid) <> ''
                """
            )
            # protein -> document, matching gene symbol + name aliases (stored patterns)
            if have_docs:
                con.execute(
                    f"""
                    INSERT INTO link_protein_document
                    WITH meta AS (
                        SELECT DISTINCT n.accession, n.gene, m.pmcid
                        FROM entity_protein_names n JOIN _doc_meta m
                          ON regexp_matches(m.blob, n.pattern)
                    ),
                    body AS (
                        SELECT n.accession, n.gene, b.pmcid,
                               sum(len(regexp_extract_all(b.blob, n.pattern))) AS n_body
                        FROM entity_protein_names n JOIN _doc_body b
                          ON regexp_matches(b.blob, n.pattern)
                        GROUP BY n.accession, n.gene, b.pmcid
                    )
                    SELECT coalesce(m.accession, b.accession), coalesce(m.gene, b.gene),
                           coalesce(m.pmcid, b.pmcid), m.pmcid IS NOT NULL,
                           coalesce(b.n_body, 0),
                           CASE WHEN m.pmcid IS NOT NULL OR coalesce(b.n_body,0) >= 2 THEN 'strong' ELSE 'weak' END,
                           {_LINK_SCORE.format(meta="m.pmcid IS NOT NULL", nbody="b.n_body")}
                    FROM meta m FULL OUTER JOIN body b
                      ON m.accession = b.accession AND m.pmcid = b.pmcid
                    """
                )

        # ---- cheminformatics + genomics arm ----
        con.execute("CREATE OR REPLACE TABLE link_drug_pubchem "
                    "(drug_norm TEXT, chembl_id TEXT, cid BIGINT, inchi_key TEXT)")
        if "pubchem_compounds" in tables and "chembl_molecules" in tables:
            con.execute(
                """
                INSERT INTO link_drug_pubchem
                SELECT DISTINCT e.drug_norm, e.chembl_id, p.cid, p.inchi_key
                FROM entity_drug_molecules e
                JOIN chembl_molecules m ON e.chembl_id = m.chembl_id
                JOIN pubchem_compounds p ON m.inchi_key = p.inchi_key
                WHERE m.inchi_key IS NOT NULL
                """
            )
        con.execute("CREATE OR REPLACE TABLE link_protein_gene "
                    "(accession TEXT, gene TEXT, ensembl_id TEXT)")
        if "ensembl_genes" in tables and have_prot:  # entity_proteins built above
            con.execute(
                """
                INSERT INTO link_protein_gene
                SELECT DISTINCT p.accession, p.gene, g.ensembl_id
                FROM entity_proteins p JOIN ensembl_genes g ON p.gene = g.gene
                """
            )

        counts = {
            t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in ("entity_drugs", "entity_drug_names", "link_drug_trial",
                      "link_drug_document", "link_trial_document",
                      "link_drug_protein", "link_protein_structure", "link_protein_document",
                      "link_drug_pubchem", "link_protein_gene")
        }
        strong = con.execute(
            "SELECT count(*) FROM link_drug_document WHERE confidence='strong'"
        ).fetchone()[0]
        print(
            f"[links]   drugs={counts['entity_drugs']} (terms={counts['entity_drug_names']})  "
            f"drug-trial={counts['link_drug_trial']}  "
            f"drug-doc={counts['link_drug_document']} ({strong} strong)  "
            f"trial-doc={counts['link_trial_document']}"
        )
        print(
            f"[links]   drug-protein={counts['link_drug_protein']}  "
            f"protein-structure={counts['link_protein_structure']}  "
            f"protein-doc={counts['link_protein_document']}"
        )
        print(
            f"[links]   drug-pubchem={counts['link_drug_pubchem']}  "
            f"protein-gene={counts['link_protein_gene']}"
        )
        return counts
    finally:
        if owns:
            con.close()


def report(con: duckdb.DuckDBPyConnection | None = None) -> None:
    owns = con is None
    con = con or connect()
    try:
        if "link_drug_trial" not in {r[0] for r in con.execute("SHOW TABLES").fetchall()}:
            print("No links built yet — run `links build`.")
            return
        print("\n========== Cross-source links ==========\n")
        rows = con.execute(
            """
            SELECT e.drug_norm, e.max_phase,
                   count(DISTINCT CASE WHEN dt.in_intervention THEN dt.nct_id END) AS trials,
                   count(DISTINCT CASE WHEN dd.confidence='strong' THEN dd.pmcid END) AS papers
            FROM entity_drugs e
            LEFT JOIN link_drug_trial dt    USING (drug_norm)
            LEFT JOIN link_drug_document dd USING (drug_norm)
            GROUP BY e.drug_norm, e.max_phase
            HAVING trials > 0 OR papers > 0
            ORDER BY (trials + papers) DESC LIMIT 12
            """
        ).fetchall()
        print(f"{'drug':16} {'phase':>5} {'trials':>6} {'papers':>6}   (strong links only)")
        print("-" * 50)
        for drug, phase, trials, papers in rows:
            print(f"{drug:16} {str(phase or '-'):>5} {trials:6} {papers:6}")
        print("\n=======================================\n")
    finally:
        if owns:
            con.close()


def explore(drug: str, con: duckdb.DuckDBPyConnection | None = None) -> None:
    """Show everything linked to one drug: its trials and its papers, by confidence."""
    owns = con is None
    con = con or connect()
    try:
        norm = _norm(drug) or drug.lower()
        ent = con.execute(
            "SELECT n_forms, max_phase, sample_name FROM entity_drugs WHERE drug_norm=?", [norm]
        ).fetchone()
        names = [r[0] for r in con.execute(
            "SELECT term FROM entity_drug_names WHERE drug_norm=? ORDER BY term", [norm]
        ).fetchall()]
        print(f"\n=== {drug}  (canonical '{norm}') ===")
        if ent:
            print(f"  ChEMBL forms: {ent[0]}   max phase: {ent[1] or '-'}   e.g. {ent[2]}")
        if names:
            print(f"  match terms: {', '.join(names)}")

        trials = con.execute(
            """
            SELECT t.nct_id, t.phases, t.status, l.in_intervention, t.title
            FROM link_drug_trial l JOIN clinical_trials t USING (nct_id)
            WHERE l.drug_norm = ? ORDER BY l.in_intervention DESC LIMIT 8
            """, [norm],
        ).fetchall()
        print(f"\nTrials ({len(trials)} shown):")
        for nct, phase, status, iv, title in trials:
            tag = "intervention" if iv else "mentioned"
            print(f"  {nct}  [{phase or '-'}/{status}] ({tag})  {(title or '')[:54]}")

        papers = con.execute(
            """
            SELECT d.pmcid, d.source, l.in_metadata, l.n_body, l.confidence, d.title
            FROM link_drug_document l JOIN documents_raw d USING (pmcid)
            WHERE l.drug_norm = ? AND l.confidence='strong'
            ORDER BY l.in_metadata DESC, l.n_body DESC LIMIT 8
            """, [norm],
        ).fetchall()
        weak = con.execute(
            "SELECT count(*) FROM link_drug_document WHERE drug_norm=? AND confidence='weak'", [norm]
        ).fetchone()[0]
        print(f"\nPapers — strong ({len(papers)} shown; {weak} weak co-mentions hidden):")
        for pmcid, source, in_meta, n_body, _conf, title in papers:
            where = "metadata" if in_meta else f"body×{n_body}"
            print(f"  {pmcid} [{source}] ({where})  {(title or '')[:52]}")

        targets = con.execute(
            "SELECT DISTINCT gene, action_type FROM link_drug_protein WHERE drug_norm=?", [norm]
        ).fetchall()
        if targets:
            print("\nTarget proteins:")
            for gene, action in targets:
                print(f"  {gene}  ({action})")
        print()
    finally:
        if owns:
            con.close()


def explore_protein(gene: str, con: duckdb.DuckDBPyConnection | None = None) -> None:
    """Show everything linked to one protein/gene: drugs, structures, papers."""
    owns = con is None
    con = con or connect()
    try:
        gnorm = gene.lower()
        ent = con.execute(
            "SELECT accession, protein_name, organism FROM entity_proteins WHERE gene_norm=?", [gnorm]
        ).fetchone()
        print(f"\n=== {gene} ===")
        if ent:
            print(f"  UniProt {ent[0]}  {ent[1]}  ({ent[2]})")

        drugs = con.execute(
            "SELECT DISTINCT drug_norm, action_type FROM link_drug_protein WHERE lower(gene)=? LIMIT 12", [gnorm]
        ).fetchall()
        print(f"\nDrugs targeting it ({len(drugs)}):")
        for drug, action in drugs:
            print(f"  {drug}  ({action})")

        structs = con.execute(
            """
            SELECT s.pdb_id, s.method, s.resolution, s.title
            FROM link_protein_structure l LEFT JOIN pdb_structures s USING (pdb_id)
            WHERE lower(l.gene)=? AND s.pdb_id IS NOT NULL
            ORDER BY s.resolution NULLS LAST LIMIT 6
            """, [gnorm],
        ).fetchall()
        n_struct = con.execute(
            "SELECT count(*) FROM link_protein_structure WHERE lower(gene)=?", [gnorm]
        ).fetchone()[0]
        print(f"\nStructures ({n_struct} referenced; {len(structs)} with metadata):")
        for pid, method, res, title in structs:
            print(f"  {pid}  [{method}, {res}A]  {(title or '')[:48]}")

        papers = con.execute(
            """
            SELECT d.pmcid, d.source, l.in_metadata, l.n_body, d.title
            FROM link_protein_document l JOIN documents_raw d USING (pmcid)
            WHERE lower(l.gene)=? AND l.confidence='strong'
            ORDER BY l.in_metadata DESC, l.n_body DESC LIMIT 6
            """, [gnorm],
        ).fetchall()
        print(f"\nPapers — strong ({len(papers)} shown):")
        for pmcid, source, in_meta, n_body, title in papers:
            where = "metadata" if in_meta else f"body×{n_body}"
            print(f"  {pmcid} [{source}] ({where})  {(title or '')[:50]}")
        print()
    finally:
        if owns:
            con.close()
