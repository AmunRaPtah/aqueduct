"""Helpers to seed a synthetic landing zone (no network)."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from aqueduct import config

_NOW = datetime(2025, 1, 1, tzinfo=timezone.utc).isoformat()


def build_jats(title: str, abstract: str, sections: list[tuple[str, str]],
               *, journal: str = "Test Journal", doi: str = "10.1/x") -> str:
    """Minimal but valid JATS-XML with title, abstract, and body <sec>s."""
    secs = "".join(f"<sec><title>{t}</title><p>{p}</p></sec>" for t, p in sections)
    return (
        "<pmc-articleset><article>"
        "<front>"
        f"<journal-meta><journal-title>{journal}</journal-title></journal-meta>"
        "<article-meta>"
        f'<article-id pub-id-type="doi">{doi}</article-id>'
        f"<title-group><article-title>{title}</article-title></title-group>"
        f"<abstract><p>{abstract}</p></abstract>"
        "</article-meta></front>"
        f"<body>{secs}</body>"
        "</article></pmc-articleset>"
    )


ATOM_ENTRY = (
    '<entry xmlns="http://www.w3.org/2005/Atom" '
    'xmlns:arxiv="http://arxiv.org/schemas/atom">'
    "<id>http://arxiv.org/abs/2501.00001v1</id>"
    "<title>A Note on Mathematical Modeling</title>"
    "<summary>We study a model of addiction dynamics.</summary>"
    "<author><name>Jane Doe</name></author>"
    "<published>2025-01-02T00:00:00Z</published>"
    '<arxiv:doi>10.9/y</arxiv:doi>'
    '<arxiv:primary_category term="math.NA"/>'
    '<category term="math.NA"/><category term="q-bio.NC"/>'
    '<link title="pdf" href="http://arxiv.org/pdf/2501.00001v1"/>'
    "</entry>"
)


def seed_document(pmcid="PMC1", *, title="Fentanyl pharmacology",
                  abstract="A study of fentanyl.", sections=None,
                  mesh=None, keywords=None, source="europepmc", doi="10.1/x"):
    """Write one document (JATS xml + manifest line) into the europepmc landing zone."""
    sections = sections or [("Introduction", "Background on opioids.")]
    d = config.raw_source_dir(source)
    xml = build_jats(title, abstract, sections)
    xmlp = d / f"{pmcid.replace(':', '_')}.xml"
    xmlp.write_text(xml, encoding="utf-8")
    rec = {
        "pmcid": pmcid, "pmid": None, "doi": doi, "title": title,
        "journal": "Test Journal", "pub_year": 2025, "authors": "Doe J",
        "source": source, "query": "test", "fetched_at": _NOW, "has_body": True,
        "abstract": abstract, "mesh": mesh, "keywords": keywords,
        "grants": None, "cited_by": 3, "xml_file": str(xmlp),
    }
    with (d / "manifest.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _write_jsonl(source: str, name: str, rows: list[dict]) -> None:
    d = config.raw_source_dir(source)
    with (d / name).open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def seed_chembl():
    _write_jsonl("chembl", "molecules.jsonl", [
        {"chembl_id": "CHEMBL_FENT", "pref_name": "FENTANYL", "max_phase": 4.0,
         "mw": 336.5, "alogp": 3.8, "ro5_violations": 0, "smiles": "CCC"},
        {"chembl_id": "CHEMBL_NLX", "pref_name": "NALOXONE HYDROCHLORIDE DIHYDRATE",
         "max_phase": 4.0, "mw": 399.9, "ro5_violations": 0},
    ])
    _write_jsonl("chembl", "synonyms.jsonl", [
        {"chembl_id": "CHEMBL_FENT", "syn_type": "TRADE_NAME", "name": "Duragesic"},
        {"chembl_id": "CHEMBL_NLX", "syn_type": "TRADE_NAME", "name": "Narcan"},
    ])
    _write_jsonl("chembl", "mechanisms.jsonl", [
        {"molecule_chembl_id": "CHEMBL_FENT", "target_chembl_id": "T_OPRM1",
         "action_type": "AGONIST", "mechanism_of_action": "Mu agonist"},
    ])


def seed_clinicaltrials():
    _write_jsonl("clinicaltrials", "trials.jsonl", [
        {"nct_id": "NCT1", "title": "Fentanyl in the ED", "status": "COMPLETED",
         "study_type": "INTERVENTIONAL", "phases": "PHASE4", "enrollment": 100,
         "conditions": "Pain", "interventions": "DRUG:Fentanyl", "lead_sponsor": "U"},
    ])


def seed_uniprot():
    _write_jsonl("uniprot", "proteins.jsonl", [
        {"accession": "P35372", "entry_name": "OPRM_HUMAN",
         "protein_name": "Mu-type opioid receptor", "gene": "OPRM1",
         "organism": "Homo sapiens", "length": 400, "function": "Receptor.",
         "pdb_ids": "5C1M; 8E0G", "chembl_target": "T_OPRM1",
         "aliases": "Mu-type opioid receptor; Mu opioid receptor; MOP; MOR1"},
    ])


def seed_pdb():
    _write_jsonl("pdb", "structures.jsonl", [
        {"pdb_id": "5C1M", "title": "Active mu-opioid receptor", "method": "X-ray",
         "resolution": 2.07},
    ])
