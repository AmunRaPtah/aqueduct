"""Orchestration for the full-text document pipeline."""

from __future__ import annotations

from . import documents
from .sources import arxiv, europepmc, openalex, patents
from .storage import connect

INGESTORS = {
    "europepmc": europepmc.ingest,
    "arxiv": arxiv.ingest,
    "openalex": openalex.ingest,
    "patents": patents.ingest,
}


def build(con=None) -> None:
    """Run store -> process -> chunk over whatever is in the landing zone."""
    owns = con is None
    con = con or connect()
    try:
        documents.store_documents(con)
        documents.build_clusters(con)   # cross-source DOI de-duplication
        documents.process_documents(con)
        documents.chunk_documents(con)
    finally:
        if owns:
            con.close()


def run(query: str, limit: int = 25, source: str = "europepmc", fulltext: bool = False) -> None:
    """Ingest from one source, build all layers, print the corpus report."""
    print(f"=== Aqueduct corpus: {source} {query!r} (limit {limit}) ===")
    kw = {"fulltext": fulltext} if source == "arxiv" else {}
    INGESTORS[source](query, limit=limit, **kw)
    con = connect()
    try:
        build(con)
        documents.report(con)
    finally:
        con.close()
    print("=== done ===")
