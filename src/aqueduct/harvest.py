"""Topic-driven harvesting — the systematic alternative to ad-hoc manual queries.

A *topics file* (JSON) lists the searches you care about per source. `harvest` runs
them all (incrementally accumulating in the landing zone), then rebuilds the corpus,
datasets, links, and semantic index. One command, repeatable, schedulable.

Topics file shape:
    {
      "documents":  {"europepmc": ["q1", "q2"], "openalex": ["q3"], "arxiv": ["q4"]},
      "structured": {"chembl": ["opioid"], "uniprot": ["opioid receptor"],
                     "ensembl": [""], "bindingdb": [""]}
    }
Empty-string queries mean "enrich from what's already landed" (pdb/ensembl/bindingdb).
List `uniprot` before pdb/ensembl/bindingdb, since those enrich UniProt accessions/genes.
"""

from __future__ import annotations

from . import corpus, datasets, embeddings, links
from .sources import (bindingdb, chembl, clinicaltrials, ensembl, pdb, pubchem,
                      uniprot)
from .storage import connect

# structured-source ingestors (document ingestors live in corpus.INGESTORS)
DATA_INGESTORS = {
    "chembl": chembl.ingest,
    "clinicaltrials": clinicaltrials.ingest,
    "uniprot": uniprot.ingest,
    "pdb": pdb.ingest,
    "pubchem": pubchem.ingest,
    "ensembl": ensembl.ingest,
    "bindingdb": bindingdb.ingest,
}


def _run(label: str, ingestors: dict, plan: dict, limit: int) -> int:
    n = 0
    for source, queries in (plan or {}).items():
        ing = ingestors.get(source)
        if ing is None:
            print(f"  ! unknown {label} source: {source}")
            continue
        for q in queries:
            try:
                ing(q, limit=limit)
                n += 1
            except Exception as e:  # noqa: BLE001 - one bad query shouldn't stop the run
                print(f"  ! {source} {q!r}: {e}")
    return n


def harvest(topics: dict, limit: int = 25, build: bool = True) -> dict:
    """Run every (source, query) in `topics`, then rebuild everything."""
    print("=== Aqueduct harvest ===")
    docs = _run("document", corpus.INGESTORS, topics.get("documents", {}), limit)
    structs = _run("structured", DATA_INGESTORS, topics.get("structured", {}), limit)
    print(f"[harvest] ran {docs} document + {structs} structured queries")
    if build:
        con = connect()
        try:
            corpus.build(con)
            datasets.build(con)
            links.build(con)
            embeddings.build_index(con)
        finally:
            con.close()
    print("=== harvest done ===")
    return {"documents": docs, "structured": structs}
