"""PubChem + Ensembl flatten + graph-link tests (offline)."""

from __future__ import annotations

import json

import seed

from aqueduct import config, datasets, links
from aqueduct.sources import ensembl, pubchem


def test_pubchem_flatten_types():
    p = {"CID": 5284596, "MolecularFormula": "C19H21NO4", "MolecularWeight": "327.4",
         "XLogP": 2.1, "InChIKey": "KEY-1", "IUPACName": "x"}
    f = pubchem._flatten(p, "naloxone")
    assert f["cid"] == 5284596 and f["mw"] == 327.4 and isinstance(f["mw"], float)
    assert f["inchi_key"] == "KEY-1"


def test_ensembl_flatten():
    d = {"id": "ENSG1", "biotype": "protein_coding", "seq_region_name": "6",
         "start": 1, "end": 9, "strand": 1, "description": "opioid receptor"}
    f = ensembl._flatten("OPRM1", d)
    assert f["gene"] == "OPRM1" and f["ensembl_id"] == "ENSG1" and f["chromosome"] == "6"


def _seed_chem(env):
    # a chembl molecule + a pubchem compound that share an InChIKey
    d = config.raw_source_dir("chembl")
    (d / "molecules.jsonl").write_text(json.dumps(
        {"chembl_id": "CHEMBL_X", "pref_name": "FENTANYL", "max_phase": 4.0,
         "inchi_key": "SHAREDKEY"}) + "\n")
    d2 = config.raw_source_dir("pubchem")
    (d2 / "compounds.jsonl").write_text(json.dumps(
        {"cid": 3345, "inchi_key": "SHAREDKEY", "molecular_formula": "C22", "mw": 336.5}) + "\n")
    d3 = config.raw_source_dir("ensembl")
    (d3 / "genes.jsonl").write_text(json.dumps(
        {"gene": "OPRM1", "ensembl_id": "ENSG00000112038", "biotype": "protein_coding",
         "chromosome": "6"}) + "\n")


def test_links_pubchem_and_gene(con, env):
    _seed_chem(env)
    seed.seed_uniprot()
    datasets.build(con)
    links.build(con)
    # drug <-> pubchem by shared InChIKey
    row = con.execute(
        "SELECT cid FROM link_drug_pubchem WHERE drug_norm='fentanyl'").fetchone()
    assert row and row[0] == 3345
    # protein <-> gene by symbol
    g = con.execute(
        "SELECT ensembl_id FROM link_protein_gene WHERE gene='OPRM1'").fetchone()
    assert g and g[0] == "ENSG00000112038"
