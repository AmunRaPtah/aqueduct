"""Command-line entrypoint: `python -m aqueduct <command>`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import (analysis, analytics, corpus, datasets, discover, documents, embeddings,
               harvest, ingest, links, pipeline, process, rag, reports, storage)
from .sources import (arxiv, bindingdb, chembl, clinicaltrials, ensembl, europepmc,
                      openalex, patents, pdb, pubchem, uniprot)

INGESTORS = {
    "europepmc": europepmc.ingest,
    "arxiv": arxiv.ingest,
    "openalex": openalex.ingest,
    "patents": patents.ingest,
}
DATA_INGESTORS = {
    "chembl": chembl.ingest,
    "clinicaltrials": clinicaltrials.ingest,
    "uniprot": uniprot.ingest,
    "pdb": pdb.ingest,
    "pubchem": pubchem.ingest,
    "ensembl": ensembl.ingest,
    "bindingdb": bindingdb.ingest,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="aqueduct", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # --- real corpus pipeline (Europe PMC full text) ---
    c_run = sub.add_parser("corpus", help="full-text corpus pipeline (Europe PMC)")
    c_sub = c_run.add_subparsers(dest="corpus_cmd", required=True)

    cr = c_sub.add_parser("run", help="fetch + build + report end-to-end")
    cr.add_argument("--query", required=True, help="query, e.g. 'CRISPR gene therapy'")
    cr.add_argument("--limit", type=int, default=25, help="max articles")
    cr.add_argument("--source", choices=list(INGESTORS), default="europepmc")
    cr.add_argument("--fulltext", action="store_true", help="arXiv: extract PDF body text")

    cf = c_sub.add_parser("fetch", help="ingest full text into the landing zone")
    cf.add_argument("--query", required=True)
    cf.add_argument("--limit", type=int, default=25)
    cf.add_argument("--source", choices=list(INGESTORS), default="europepmc")
    cf.add_argument("--fulltext", action="store_true", help="arXiv: extract PDF body text")

    c_sub.add_parser("build", help="store + process + chunk the landing zone")
    c_sub.add_parser("report", help="print the corpus report")

    cq = c_sub.add_parser("search", help="lexical search over chunks")
    cq.add_argument("term")
    cq.add_argument("-k", type=int, default=8)

    ci = c_sub.add_parser("index", help="build the semantic index over chunks")
    ci.add_argument("--backend", choices=["auto", "lsa", "st"], default="auto",
                    help="auto (st if installed, else lsa), or force lsa/st")
    ci.add_argument("--dims", type=int, default=128, help="LSA dimensions")
    ci.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model")
    cs = c_sub.add_parser("semantic", help="semantic search over chunks")
    cs.add_argument("query")
    cs.add_argument("-k", type=int, default=8)
    cs.add_argument("--json", action="store_true", help="emit JSON (for programmatic use)")

    # --- structured datasets (ChEMBL, ...) ---
    d_grp = sub.add_parser("data", help="structured datasets (ChEMBL, clinical, ...)")
    d_sub = d_grp.add_subparsers(dest="data_cmd", required=True)
    df = d_sub.add_parser("fetch", help="ingest a structured source")
    df.add_argument("--source", choices=list(DATA_INGESTORS), default="chembl")
    df.add_argument("--query", default="", help="source query (ignored for pdb)")
    df.add_argument("--limit", type=int, default=100)
    d_sub.add_parser("build", help="load landing-zone records into typed tables")
    d_sub.add_parser("report", help="print the structured-data report")

    # --- analysis + reporting (DuckDB facts + DeepSeek interpretation) ---
    rp = sub.add_parser("report", help="grounded analysis report (facts + DeepSeek)")
    rp.add_argument("--topic", default=None, help="focus topic (also pulls relevant excerpts)")
    rp.add_argument("--agent", action="store_true",
                    help="bounded agentic Claude-on-DeepSeek deep analysis (more tokens)")
    rp.add_argument("--model", default="pro", help="pro (default) or flash")
    rp.add_argument("--email", action="store_true", help="email the report (Resend)")
    rp.add_argument("--to", default=None, help="recipient (default AQUEDUCT_EMAIL_TO)")
    fp = sub.add_parser("facts", help="print the computed metrics (no LLM, no tokens)")
    fp.add_argument("--json", action="store_true", help="emit JSON")

    # --- RAG retrieval surface for other systems (e.g. the Pardalos agent) ---
    rg = sub.add_parser("rag", help="JSON RAG context (chunks + citations + graph) for a query")
    rg.add_argument("query")
    rg.add_argument("-k", type=int, default=8)
    rg.add_argument("--no-graph", action="store_true", help="skip graph context")
    rg.add_argument("--min-score", type=float, default=0.0, help="drop matches below this score")
    rg.add_argument("--source", action="append", help="restrict to source(s); repeatable")
    rg.add_argument("--section", action="append", help="restrict to sec_type(s) e.g. abstract")

    sv = sub.add_parser("serve", help="run the HTTP retrieval API (for remote consumers)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8800)

    # --- topic-driven harvest (the systematic, list-based trigger) ---
    hv = sub.add_parser("harvest", help="run all queries from a topics file, then build")
    hv.add_argument("--topics", default="topics.json", help="path to a topics JSON file")
    hv.add_argument("--limit", type=int, default=25, help="max results per query")
    hv.add_argument("--no-build", action="store_true", help="ingest only; skip rebuild")

    # --- cross-source entity links (drug <-> trial <-> paper) ---
    l_grp = sub.add_parser("links", help="cross-source entity links")
    l_sub = l_grp.add_subparsers(dest="links_cmd", required=True)
    l_sub.add_parser("build", help="build the drug/trial/paper/protein link graph")
    l_sub.add_parser("report", help="best-connected drugs across sources")
    le = l_sub.add_parser("explore", help="show trials + papers + targets linked to a drug")
    le.add_argument("drug")
    lp = l_sub.add_parser("protein", help="show drugs + structures + papers linked to a gene")
    lp.add_argument("gene")
    ld = l_sub.add_parser("discover", help="semantic-over-graph: papers about a drug's biology")
    ld.add_argument("drug")
    ld.add_argument("-k", type=int, default=8)
    ldp = l_sub.add_parser("discover-protein", help="semantic-over-graph: papers about a gene's biology")
    ldp.add_argument("gene")
    ldp.add_argument("-k", type=int, default=8)

    # --- synthetic events demo (original scaffold) ---
    p_run = sub.add_parser("run", help="[demo] synthetic events pipeline")
    p_run.add_argument("--events", type=int, default=2000)
    p_run.add_argument("--seed", type=int, default=42)
    p_ing = sub.add_parser("ingest", help="[demo] ingest synthetic events")
    p_ing.add_argument("--events", type=int, default=2000)
    p_ing.add_argument("--seed", type=int, default=42)
    sub.add_parser("store", help="[demo] load events into bronze")
    sub.add_parser("process", help="[demo] build event silver + gold")
    sub.add_parser("query", help="[demo] print the events report")

    args = parser.parse_args(argv)

    if args.command == "corpus":
        if args.corpus_cmd == "run":
            corpus.run(args.query, limit=args.limit, source=args.source, fulltext=args.fulltext)
        elif args.corpus_cmd == "fetch":
            kw = {"fulltext": args.fulltext} if args.source == "arxiv" else {}
            INGESTORS[args.source](args.query, limit=args.limit, **kw)
        elif args.corpus_cmd == "build":
            corpus.build()
        elif args.corpus_cmd == "report":
            documents.report()
        elif args.corpus_cmd == "search":
            documents.search(args.term, k=args.k)
        elif args.corpus_cmd == "index":
            embeddings.build_index(backend=args.backend, dims=args.dims, model=args.model)
        elif args.corpus_cmd == "semantic":
            if args.json:
                print(json.dumps(rag.retrieve(args.query, k=args.k, graph=False), indent=2))
            else:
                embeddings.semantic_search(args.query, k=args.k)
    elif args.command == "report":
        reports.generate(args.topic, agent=args.agent, model=args.model,
                         email=args.email, to=args.to)
    elif args.command == "facts":
        print(json.dumps(analysis.facts(), indent=2) if args.json else analysis.facts_sheet())
    elif args.command == "rag":
        print(json.dumps(rag.retrieve(
            args.query, k=args.k, graph=not args.no_graph, min_score=args.min_score,
            sources=args.source, sec_types=args.section), indent=2))
    elif args.command == "serve":
        from . import server
        server.serve(host=args.host, port=args.port)
    elif args.command == "harvest":
        topics = json.loads(Path(args.topics).read_text())
        harvest.harvest(topics, limit=args.limit, build=not args.no_build)
    elif args.command == "data":
        if args.data_cmd == "fetch":
            DATA_INGESTORS[args.source](args.query, limit=args.limit)
        elif args.data_cmd == "build":
            datasets.build()
        elif args.data_cmd == "report":
            datasets.report()
    elif args.command == "links":
        if args.links_cmd == "build":
            links.build()
        elif args.links_cmd == "report":
            links.report()
        elif args.links_cmd == "explore":
            links.explore(args.drug)
        elif args.links_cmd == "protein":
            links.explore_protein(args.gene)
        elif args.links_cmd == "discover":
            discover.drug(args.drug, k=args.k)
        elif args.links_cmd == "discover-protein":
            discover.protein(args.gene, k=args.k)
    elif args.command == "run":
        pipeline.run(n_events=args.events, seed=args.seed)
    elif args.command == "ingest":
        ingest.ingest(n_events=args.events, seed=args.seed)
    elif args.command == "store":
        storage.store()
    elif args.command == "process":
        process.process()
    elif args.command == "query":
        analytics.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
