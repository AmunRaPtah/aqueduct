"""Structured-mode pipeline (records, not documents).

Land flat JSONL from a structured source, then load it into a typed DuckDB table
with DuckDB's native JSON reader. Each source gets its own table (separate tables;
cross-source links are a later step). Mirrors the document pipeline's bronze->silver
shape, minus full-text parsing/chunking.
"""

from __future__ import annotations

import duckdb

from . import config
from .storage import connect

# source -> list of (landing JSONL file, target table); a source may load >1 table.
DATASETS = {
    "chembl": [
        ("chembl/molecules.jsonl", "chembl_molecules"),
        ("chembl/synonyms.jsonl", "chembl_synonyms"),
        ("chembl/mechanisms.jsonl", "chembl_mechanisms"),
    ],
    "clinicaltrials": [("clinicaltrials/trials.jsonl", "clinical_trials")],
    "uniprot": [("uniprot/proteins.jsonl", "uniprot_proteins")],
    "pdb": [("pdb/structures.jsonl", "pdb_structures")],
    "pubchem": [("pubchem/compounds.jsonl", "pubchem_compounds")],
    "ensembl": [("ensembl/genes.jsonl", "ensembl_genes")],
    "bindingdb": [("bindingdb/affinities.jsonl", "binding_affinities")],
}


def _load_one(con: duckdb.DuckDBPyConnection, source: str) -> int:
    total = 0
    for rel, table in DATASETS[source]:
        path = config.RAW_DIR / rel
        if not path.exists():
            continue
        con.execute(f"DROP TABLE IF EXISTS {table}")
        con.execute(
            f"CREATE TABLE {table} AS "
            f"SELECT * FROM read_json_auto('{path}', format='newline_delimited')"
        )
        rows = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"[load]    {rows} rows -> {table}")
        total += rows
    return total


def build(con: duckdb.DuckDBPyConnection | None = None) -> dict[str, int]:
    """Load every structured source present in the landing zone into its table."""
    owns = con is None
    con = con or connect()
    try:
        counts = {s: _load_one(con, s) for s in DATASETS}
        if not any(counts.values()):
            print("[load]    no structured datasets in landing zone")
        return counts
    finally:
        if owns:
            con.close()


# Backwards-compatible single-source loader.
def load_chembl(con: duckdb.DuckDBPyConnection | None = None) -> int:
    owns = con is None
    con = con or connect()
    try:
        return _load_one(con, "chembl")
    finally:
        if owns:
            con.close()


def _has_table(con, name: str) -> bool:
    return name in {r[0] for r in con.execute("SHOW TABLES").fetchall()}


def report(con: duckdb.DuckDBPyConnection | None = None) -> None:
    owns = con is None
    con = con or connect()
    try:
        print("\n========== Structured datasets ==========\n")
        shown = False

        if _has_table(con, "chembl_molecules"):
            shown = True
            n = con.execute("SELECT count(*) FROM chembl_molecules").fetchone()[0]
            print(f"chembl_molecules: {n} rows")
            print("  -- by max clinical phase --")
            for phase, c in con.execute(
                "SELECT max_phase, count(*) FROM chembl_molecules "
                "GROUP BY max_phase ORDER BY max_phase DESC NULLS LAST"
            ).fetchall():
                print(f"     phase {phase}: {c}")
            print("  -- sample approved/known drugs --")
            for cid, name, ph in con.execute(
                "SELECT chembl_id, pref_name, max_phase FROM chembl_molecules "
                "WHERE pref_name IS NOT NULL ORDER BY max_phase DESC NULLS LAST LIMIT 6"
            ).fetchall():
                print(f"     {cid:14} {str(name)[:34]:34} phase={ph}")
            print()

        if _has_table(con, "clinical_trials"):
            shown = True
            n = con.execute("SELECT count(*) FROM clinical_trials").fetchone()[0]
            print(f"clinical_trials: {n} rows")
            print("  -- by status --")
            for status, c in con.execute(
                "SELECT status, count(*) FROM clinical_trials "
                "GROUP BY status ORDER BY count(*) DESC LIMIT 6"
            ).fetchall():
                print(f"     {str(status):24} {c}")
            print("  -- by phase --")
            for phase, c in con.execute(
                "SELECT phases, count(*) FROM clinical_trials "
                "GROUP BY phases ORDER BY count(*) DESC LIMIT 6"
            ).fetchall():
                print(f"     {str(phase):24} {c}")
            print()

        if _has_table(con, "uniprot_proteins"):
            shown = True
            n = con.execute("SELECT count(*) FROM uniprot_proteins").fetchone()[0]
            ns = con.execute(
                "SELECT count(*) FROM pdb_structures").fetchone()[0] if _has_table(con, "pdb_structures") else 0
            print(f"uniprot_proteins: {n} rows   pdb_structures: {ns} rows")
            for gene, org, ln, chembl in con.execute(
                "SELECT gene, organism, length, chembl_target FROM uniprot_proteins "
                "WHERE gene IS NOT NULL LIMIT 6"
            ).fetchall():
                print(f"     {str(gene):10} {str(org)[:22]:22} {ln}aa  chembl={chembl}")
            print()

        if _has_table(con, "pubchem_compounds"):
            shown = True
            n = con.execute("SELECT count(*) FROM pubchem_compounds").fetchone()[0]
            print(f"pubchem_compounds: {n} rows")
            for cid, formula, mw in con.execute(
                "SELECT cid, molecular_formula, round(mw,1) FROM pubchem_compounds LIMIT 5"
            ).fetchall():
                print(f"     CID {cid}  {str(formula):16} {mw}")
            print()

        if _has_table(con, "ensembl_genes"):
            shown = True
            n = con.execute("SELECT count(*) FROM ensembl_genes").fetchone()[0]
            print(f"ensembl_genes: {n} rows")
            for gene, eid, chrom, bt in con.execute(
                "SELECT gene, ensembl_id, chromosome, biotype FROM ensembl_genes LIMIT 5"
            ).fetchall():
                print(f"     {str(gene):10} {eid}  chr{chrom}  {bt}")
            print()

        if _has_table(con, "binding_affinities"):
            shown = True
            n = con.execute("SELECT count(*) FROM binding_affinities").fetchone()[0]
            print(f"binding_affinities: {n} rows")
            for at, c, best in con.execute(
                "SELECT affinity_type, count(*), round(min(affinity_nm),1) "
                "FROM binding_affinities GROUP BY affinity_type ORDER BY count(*) DESC LIMIT 5"
            ).fetchall():
                print(f"     {str(at):8} {c:5}  best={best} nM")
            print()

        if not shown:
            print("No structured datasets loaded yet.")
        print("=========================================\n")
    finally:
        if owns:
            con.close()
