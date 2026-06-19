"""Cross-source link graph tests — the heart of the entity-resolution logic."""

from __future__ import annotations

import seed

from aqueduct import corpus, datasets, links


def _full_graph(con):
    # literature that names the drug (metadata + body) and the gene symbol (body x2)
    seed.seed_document(
        "PMC1", title="Fentanyl and the mu receptor",
        abstract="Fentanyl is a mu-opioid agonist.",
        sections=[("Pharmacology",
                   "Duragesic delivers fentanyl. OPRM1 mediates analgesia; OPRM1 is the target.")],
    )
    corpus.build(con)
    seed.seed_chembl()
    seed.seed_clinicaltrials()
    seed.seed_uniprot()
    seed.seed_pdb()
    datasets.build(con)
    return links.build(con)


def test_entity_resolution_includes_synonyms(con):
    _full_graph(con)
    terms = {r[0] for r in con.execute(
        "SELECT term FROM entity_drug_names WHERE drug_norm='fentanyl'").fetchall()}
    assert {"fentanyl", "duragesic"} <= terms
    # salt forms collapse to one canonical entity
    assert con.execute(
        "SELECT count(*) FROM entity_drugs WHERE drug_norm='naloxone'").fetchone()[0] == 1


def test_drug_trial_link_flags_intervention(con):
    _full_graph(con)
    row = con.execute(
        "SELECT in_intervention FROM link_drug_trial WHERE drug_norm='fentanyl' AND nct_id='NCT1'"
    ).fetchone()
    assert row is not None and row[0] is True


def test_drug_document_confidence(con):
    _full_graph(con)
    # fentanyl appears in title/abstract -> strong
    conf = con.execute(
        "SELECT confidence FROM link_drug_document WHERE drug_norm='fentanyl' AND pmcid='PMC1'"
    ).fetchone()[0]
    assert conf == "strong"


def test_drug_protein_bridge(con):
    _full_graph(con)
    row = con.execute(
        "SELECT accession, gene, action_type FROM link_drug_protein WHERE drug_norm='fentanyl'"
    ).fetchone()
    assert row == ("P35372", "OPRM1", "AGONIST")


def test_protein_structure_and_document_links(con):
    _full_graph(con)
    pdbs = {r[0] for r in con.execute(
        "SELECT pdb_id FROM link_protein_structure WHERE gene='OPRM1'").fetchall()}
    assert {"5C1M", "8E0G"} <= pdbs
    # OPRM1 mentioned twice in body -> strong protein-document link
    conf = con.execute(
        "SELECT confidence FROM link_protein_document WHERE gene='OPRM1' AND pmcid='PMC1'"
    ).fetchone()[0]
    assert conf == "strong"


def test_synonym_only_mention_links_drug_to_doc(con):
    """A paper that says only the trade name still links to the canonical drug."""
    seed.seed_document("PMC9", title="Pain management review",
                       abstract="We review Duragesic patches for chronic pain.")
    corpus.build(con)
    seed.seed_chembl()
    datasets.build(con)
    links.build(con)
    hit = con.execute(
        "SELECT count(*) FROM link_drug_document WHERE drug_norm='fentanyl' AND pmcid='PMC9'"
    ).fetchone()[0]
    assert hit == 1
