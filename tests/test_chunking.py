"""Chunk metadata (#2) + sentence-aware, configurable chunking."""

from __future__ import annotations

import seed

from aqueduct import corpus, documents, embeddings, jats, rag


def test_section_kind_classifies_imrad():
    assert documents.section_kind("title", None) == "title"
    assert documents.section_kind("abstract", "abstract") == "abstract"
    assert documents.section_kind("body", "2. Materials and Methods") == "methods"
    assert documents.section_kind("body", "Results and discussion") == "results"
    assert documents.section_kind("body", "1. Introduction") == "introduction"
    assert documents.section_kind("body", "Random heading") == "body"


def test_jats_counts_figures_and_tables():
    xml = (
        "<pmc-articleset><article><front>"
        "<article-meta><title-group><article-title>T</article-title></title-group>"
        "<abstract><p>An abstract.</p></abstract></article-meta></front>"
        "<body><sec><title>Results</title><p>We saw an effect.</p>"
        "<fig><caption><p>Fig 1</p></caption></fig>"
        "<table-wrap><label>Table 1</label></table-wrap>"
        "<fig><caption><p>Fig 2</p></caption></fig></sec></body>"
        "</article></pmc-articleset>"
    )
    secs = jats.parse_jats(xml)["sections"]
    body = [s for s in secs if s["sec_type"] == "body"][0]
    assert body["n_figures"] == 2 and body["n_tables"] == 1


def test_sentence_aware_chunks_split_on_boundaries():
    text = " ".join(f"Sentence number {i} has several words in it." for i in range(20))
    chunks = list(documents._chunk_sentences(text, size=20, overlap=6))
    assert len(chunks) > 1
    # every chunk (except possibly the last) ends at a sentence boundary
    for ch in chunks[:-1]:
        assert " ".join(ch).rstrip().endswith(".")
    # no chunk blows far past the target size
    assert all(len(ch) <= 20 for ch in chunks)


def test_long_sentence_falls_back_to_hard_split():
    text = "word " * 50  # one 50-word "sentence" with no boundaries
    chunks = list(documents._chunk_sentences(text.strip(), size=20, overlap=5))
    assert chunks and all(len(ch) <= 20 for ch in chunks)


def test_chunks_carry_structural_metadata(con, env):
    seed.seed_document(
        "PMC1", title="Opioid study",
        abstract="Fentanyl is a mu-opioid agonist.",
        sections=[("Methods", "We dosed subjects. " * 8),
                  ("Results", "Analgesia increased. " * 8)],
    )
    corpus.build(con)
    rows = con.execute(
        "SELECT DISTINCT sec_kind, is_methods, is_results FROM doc_chunks ORDER BY sec_kind"
    ).fetchall()
    kinds = {r[0] for r in rows}
    assert {"methods", "results"} <= kinds
    m = con.execute("SELECT is_methods, is_results FROM doc_chunks WHERE sec_kind='methods' LIMIT 1").fetchone()
    assert m == (True, False)


def test_rag_exposes_and_filters_by_kind(con, env):
    seed.seed_document(
        "PMC1", title="Opioid receptor study",
        abstract="The mu-opioid receptor mediates analgesia.",
        sections=[("Methods", "We measured binding affinity of the agonist. " * 6),
                  ("Results", "The agonist reduced pain via the receptor. " * 6)],
    )
    corpus.build(con)
    embeddings.build_index(con, backend="lsa", dims=8)
    out = rag.retrieve("agonist analgesia receptor", k=8, con=con, kinds=["results"])
    assert out["n"] >= 1
    assert all(c["sec_kind"] == "results" for c in out["chunks"])
    assert {"sec_kind", "is_methods", "is_results", "n_figures", "n_tables"} <= set(out["chunks"][0])
