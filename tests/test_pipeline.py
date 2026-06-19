"""Document pipeline integration tests (seeded landing zone, temp warehouse)."""

from __future__ import annotations

import seed

from aqueduct import corpus, documents


def test_document_pipeline_end_to_end(con):
    seed.seed_document(
        "PMC1", title="Fentanyl study",
        abstract="Fentanyl is a potent opioid agonist.",
        sections=[("Intro", "Fentanyl " + "binds receptors. " * 60)],
    )
    seed.seed_document("PMC2", title="Naloxone reversal",
                       abstract="Naloxone reverses overdose.")

    assert documents.store_documents(con) == 2
    assert con.execute("SELECT count(*) FROM documents_raw").fetchone()[0] == 2
    # metadata is preserved alongside full text
    cited = con.execute("SELECT cited_by FROM documents_raw WHERE pmcid='PMC1'").fetchone()[0]
    assert cited == 3

    n_sec = documents.process_documents(con)
    assert n_sec >= 4  # 2 docs x (title + abstract + body)
    n_chunks = documents.chunk_documents(con)
    assert n_chunks > 0
    # chunks only come from abstract/body, never the bare title
    assert con.execute(
        "SELECT count(*) FROM doc_chunks WHERE sec_type='title'"
    ).fetchone()[0] == 0


def test_corpus_build_is_idempotent(con):
    seed.seed_document("PMC1")
    corpus.build(con)
    first = con.execute("SELECT count(*) FROM documents_raw").fetchone()[0]
    corpus.build(con)
    assert con.execute("SELECT count(*) FROM documents_raw").fetchone()[0] == first
