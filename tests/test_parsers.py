"""Unit tests for parsers, flatteners, and pure helpers (no network, no DB)."""

from __future__ import annotations

import seed

from aqueduct import documents, jats, links
from aqueduct.sources import arxiv, chembl, clinicaltrials, uniprot


# ---- JATS ----
def test_jats_extracts_title_abstract_and_nested_sections():
    xml = seed.build_jats(
        "Mu Receptor Review", "An abstract about fentanyl.",
        [("1. Intro", "Opioids bind receptors."),
         ("2. Methods", "We measured binding.")],
    )
    out = jats.parse_jats(xml)
    assert out["meta"]["title"] == "Mu Receptor Review"
    assert out["meta"]["journal"] == "Test Journal"
    assert out["meta"]["doi"] == "10.1/x"
    types = [s["sec_type"] for s in out["sections"]]
    assert types[:2] == ["title", "abstract"]
    body = [s for s in out["sections"] if s["sec_type"] == "body"]
    assert {s["sec_title"] for s in body} == {"1. Intro", "2. Methods"}


def test_jats_bad_xml_is_safe():
    assert jats.parse_jats("<not valid")["sections"] == []


# ---- arXiv Atom ----
def test_arxiv_parse_atom():
    out = arxiv.parse_atom(seed.ATOM_ENTRY)
    assert out["meta"]["arxiv_id"] == "2501.00001v1"
    assert out["meta"]["doi"] == "10.9/y"
    assert "math.NA" in out["meta"]["categories"]
    assert [s["sec_type"] for s in out["sections"]] == ["title", "abstract"]


# ---- ChEMBL flatten / typing / synonyms ----
def test_chembl_flatten_coerces_stringy_numerics():
    m = {
        "molecule_chembl_id": "CHEMBL1", "pref_name": "ASPIRIN", "max_phase": "4",
        "molecule_properties": {"full_mwt": "180.16", "num_ro5_violations": "0", "alogp": "1.2"},
        "molecule_structures": {"canonical_smiles": "CC", "standard_inchi_key": "KEY"},
    }
    flat = chembl._flatten(m)
    assert flat["mw"] == 180.16 and isinstance(flat["mw"], float)
    assert flat["max_phase"] == 4.0
    assert flat["ro5_violations"] == 0 and isinstance(flat["ro5_violations"], int)
    assert flat["smiles"] == "CC"


def test_chembl_synonyms_whitelist():
    m = {"molecule_chembl_id": "CHEMBL1", "molecule_synonyms": [
        {"syn_type": "TRADE_NAME", "molecule_synonym": "Bayer"},
        {"syn_type": "RESEARCH_CODE", "molecule_synonym": "NSC-1"},
    ]}
    names = {s["name"] for s in chembl._synonyms(m)}
    assert names == {"Bayer"}


# ---- ClinicalTrials flatten ----
def test_clinicaltrials_flatten():
    study = {"protocolSection": {
        "identificationModule": {"nctId": "NCT9", "briefTitle": "T"},
        "statusModule": {"overallStatus": "RECRUITING"},
        "designModule": {"phases": ["PHASE2"], "studyType": "INTERVENTIONAL",
                         "enrollmentInfo": {"count": 50}},
        "conditionsModule": {"conditions": ["Pain", "Addiction"]},
        "armsInterventionsModule": {"interventions": [{"type": "DRUG", "name": "Methadone"}]},
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Org"}},
    }}
    f = clinicaltrials._flatten(study)
    assert f["nct_id"] == "NCT9" and f["enrollment"] == 50
    assert f["phases"] == "PHASE2"
    assert "Addiction" in f["conditions"]
    assert "DRUG:Methadone" in f["interventions"]


# ---- UniProt flatten ----
def test_uniprot_flatten_extracts_xrefs():
    r = {
        "primaryAccession": "P1", "uniProtkbId": "X_HUMAN",
        "proteinDescription": {"recommendedName": {"fullName": {"value": "Receptor"}}},
        "genes": [{"geneName": {"value": "OPRM1"}}],
        "organism": {"scientificName": "Homo sapiens"},
        "sequence": {"length": 400},
        "comments": [{"commentType": "FUNCTION", "texts": [{"value": "Binds."}]}],
        "uniProtKBCrossReferences": [
            {"database": "PDB", "id": "5C1M"}, {"database": "PDB", "id": "8E0G"},
            {"database": "ChEMBL", "id": "T_OPRM1"},
        ],
    }
    f = uniprot._flatten(r)
    assert f["accession"] == "P1" and f["gene"] == "OPRM1"
    assert f["pdb_ids"] == "5C1M; 8E0G"
    assert f["chembl_target"] == "T_OPRM1"


# ---- normalization + chunking ----
def test_norm_strips_salts_and_filters():
    assert links._norm("NALOXONE HYDROCHLORIDE DIHYDRATE") == "naloxone"
    assert links._norm("water") is None          # stoplisted
    assert links._norm("abc") is None             # too short
    assert links._term("Duragesic") == "duragesic"
    assert links._term("Liquid pred") is None     # multiword
    assert links._term("NSC-10023") is None       # coded / short token


def test_chunk_windows_overlap():
    words = [f"w{i}" for i in range(500)]
    chunks = list(documents._chunk(words, size=220, overlap=40))
    assert all(len(c) <= 220 for c in chunks)
    assert chunks[0][-40:] == chunks[1][:40]      # overlap region matches
    assert len(chunks) >= 2
