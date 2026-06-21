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

Query-version tracking: each (source, query) run is stamped in `data/harvest_state.json`
(last-run time + run count + last outcome), so a scheduler can tell what's been
refreshed and when, and `stale_queries()` can surface searches gone stale.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from . import config, corpus, datasets, embeddings, links, obs, validate
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


def _state_path():
    return config.DATA_DIR / "harvest_state.json"


def load_state() -> dict:
    """Per-(source, query) harvest history: {"<source>\\t<query>": {runs, last_run, last_ok}}."""
    p = _state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _state_path().write_text(json.dumps(state, indent=2, sort_keys=True))


def stale_queries(state: dict | None = None, *, days: float = 7.0,
                  now: datetime | None = None) -> list[tuple[str, str, float]]:
    """Queries last refreshed more than `days` ago: [(source, query, age_days), ...]."""
    state = load_state() if state is None else state
    now = now or datetime.now(timezone.utc)
    out = []
    for key, rec in state.items():
        last = rec.get("last_run")
        if not last:
            continue
        try:
            age = (now - datetime.fromisoformat(last)).total_seconds() / 86400.0
        except ValueError:
            continue
        if age >= days:
            source, _, query = key.partition("\t")
            out.append((source, query, round(age, 2)))
    return sorted(out, key=lambda t: -t[2])


def _run(label: str, ingestors: dict, plan: dict, limit: int, state: dict, now: str) -> int:
    n = 0
    for source, queries in (plan or {}).items():
        ing = ingestors.get(source)
        if ing is None:
            print(f"  ! unknown {label} source: {source}")
            obs.log("harvest.unknown_source", kind=label, source=source)
            continue
        for q in queries:
            key = f"{source}\t{q}"
            rec = state.setdefault(key, {"runs": 0, "last_run": None, "last_ok": None})
            try:
                ing(q, limit=limit)
                rec["last_ok"] = True
                n += 1
                obs.log("harvest.query", kind=label, source=source, query=q, ok=True)
            except Exception as e:  # noqa: BLE001 - one bad query shouldn't stop the run
                print(f"  ! {source} {q!r}: {e}")
                rec["last_ok"] = False
                obs.log("harvest.query", kind=label, source=source, query=q,
                        ok=False, error_type=type(e).__name__, error=str(e))
            rec["runs"] += 1
            rec["last_run"] = now
    return n


def harvest(topics: dict, limit: int = 25, build: bool = True) -> dict:
    """Run every (source, query) in `topics`, then rebuild everything."""
    print("=== Aqueduct harvest ===")
    state = load_state()
    now = datetime.now(timezone.utc).isoformat()
    with obs.span("harvest", limit=limit):
        docs = _run("document", corpus.INGESTORS, topics.get("documents", {}), limit, state, now)
        structs = _run("structured", DATA_INGESTORS, topics.get("structured", {}), limit, state, now)
        print(f"[harvest] ran {docs} document + {structs} structured queries")
        _save_state(state)
        if build:
            con = connect()
            try:
                corpus.build(con)
                datasets.build(con)
                links.build(con)
                embeddings.build_index(con)
                validate.validate(con)
            finally:
                con.close()
    print("=== harvest done ===")
    return {"documents": docs, "structured": structs}
