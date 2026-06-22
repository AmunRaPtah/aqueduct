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

import inspect
import json
import os
from datetime import datetime, timezone

from pathlib import Path

from . import config, corpus, datasets, embeddings, links, obs, suggest, validate
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


def load_topics(path: str | Path) -> dict:
    """Load a curated topics file and merge in its auto-generated sibling.

    `suggest` writes proposals to `<path>.generated.json`; harvest runs the union of
    both so the self-expanding queries take effect without hand-editing `topics.json`.
    Curated queries come first and win on de-duplication.
    """
    curated = json.loads(Path(path).read_text())
    generated = suggest._load(suggest.generated_path(path))
    if not generated:
        return curated
    merged = {k: v for k, v in curated.items()}
    for kind in ("documents", "structured"):
        out = {s: list(qs) for s, qs in (curated.get(kind) or {}).items()}
        for source, qs in (generated.get(kind) or {}).items():
            seen = {q.lower() for q in out.get(source, [])}
            for q in qs:
                if q.lower() not in seen:
                    out.setdefault(source, []).append(q)
                    seen.add(q.lower())
        if out:
            merged[kind] = out
    return merged


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


def _accepts_cursor(fn) -> bool:
    """True if `fn` takes a `cursor` kwarg — i.e. it paginates across runs.

    Lets `_run` stay generic: paginating document ingestors get their resume cursor
    threaded in/out, while structured ingestors (and test mocks) are called as before.
    """
    try:
        return "cursor" in inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins / un-introspectable callables
        return False


def _run(label: str, ingestors: dict, plan: dict, limit: int, state: dict, now: str) -> int:
    n = 0
    for source, queries in (plan or {}).items():
        ing = ingestors.get(source)
        if ing is None:
            print(f"  ! unknown {label} source: {source}")
            obs.log("harvest.unknown_source", kind=label, source=source)
            continue
        paged = _accepts_cursor(ing)
        for q in queries:
            key = f"{source}\t{q}"
            rec = state.setdefault(key, {"runs": 0, "last_run": None, "last_ok": None})
            try:
                if paged:
                    # resume where the last run stopped; persist where this one stopped
                    result = ing(q, limit=limit, cursor=rec.get("cursor"))
                    if isinstance(result, tuple) and len(result) == 2:
                        rec["cursor"] = result[1]
                else:
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


def harvest(topics: dict, limit: int = 25, build: bool = True,
            topics_path: str | Path | None = None) -> dict:
    """Run every (source, query) in `topics`, then rebuild everything.

    After the rebuild, mine the fresh corpus for new queries and append them to the
    generated topics file (`topics_path` must be set for this self-expansion step) —
    so the next harvest searches further than this one did.
    """
    print("=== Aqueduct harvest ===")
    state = load_state()
    now = datetime.now(timezone.utc).isoformat()
    suggested: list[dict] = []
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
                # Backend is env-selectable so an unattended, memory-tight box can pin
                # the lean keyless LSA path (~400 MB, ~90 s) instead of the heavier
                # 'auto' -> sentence-transformers default (~1.5 GB, slow full re-embeds).
                embeddings.build_index(con, backend=os.environ.get("AQUEDUCT_EMBED_BACKEND", "auto"))
                validate.validate(con)
                if topics_path is not None:
                    try:
                        suggested = suggest.generate(con, topics_path)
                        if suggested:
                            print(f"[suggest] +{len(suggested)} new queries from corpus insights "
                                  + "-> " + suggest.generated_path(topics_path).name)
                            for c in suggested:
                                print(f"          {c['source']}: {c['query']!r}  ({c['reason']})")
                    except Exception as e:  # noqa: BLE001 - suggestion is best-effort, never fail a harvest
                        print(f"[suggest] skipped ({type(e).__name__}: {e})")
                        obs.log("suggest.error", error_type=type(e).__name__, error=str(e))
            finally:
                con.close()
    print("=== harvest done ===")
    return {"documents": docs, "structured": structs, "suggested": len(suggested)}
