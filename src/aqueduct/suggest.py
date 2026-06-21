"""Self-expanding watchlist — turn corpus insights into the next round of queries.

A fixed `topics.json` stalls: once its searches saturate, every harvest lands
"+0 new" and the corpus stops growing. This module closes the loop. It reads the
analysis layer's own signals (`analysis.facts`) and proposes new searches from them:

- **Undrugged but structurally-studied targets** (`asymmetries`) — a gene with many
  solved structures yet little literature is exactly where to pull more papers.
- **Targets without drugs / drugs without papers** (`gaps`) — entities the corpus
  knows about but hasn't connected; querying them fills the whitespace.
- **Frequent MeSH terms & OpenAlex concepts** (`topics`) not yet on the watchlist —
  the corpus is telling us what it's about; follow the density.

Proposals are de-duplicated against everything already covered (curated *and*
previously generated), ranked by signal strength, capped per source and in total,
and written — with provenance — to a sibling `topics.generated.json`. The curated
`topics.json` is never touched. `harvest` merges both, so the next run picks up the
new queries automatically: corpus -> insights -> queries -> corpus.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from . import analysis

# Generic MeSH headings / OpenAlex concepts that are true of almost any paper —
# promoting them as queries would just pull noise, so they never become searches.
_GENERIC = {
    "humans", "animals", "male", "female", "adult", "aged", "middle aged",
    "young adult", "child", "adolescent", "infant", "mice", "rats", "rats, sprague-dawley",
    "mice, inbred c57bl", "cells, cultured", "treatment outcome", "time factors",
    "reproducibility of results", "models, molecular", "models, biological",
    "models, theoretical", "molecular structure", "molecular sequence data",
    "amino acid sequence", "base sequence", "biology", "chemistry", "medicine",
    "physics", "mathematics", "computer science", "biochemistry", "genetics",
    "molecular biology", "materials science", "internal medicine", "pathology",
    "pharmacology", "engineering", "psychology", "artificial intelligence",
}

_WORD = re.compile(r"[a-z0-9][a-z0-9+-]{1,}")


def generated_path(topics_path: str | Path) -> Path:
    """The sibling file auto-generated queries are written to."""
    p = Path(topics_path)
    return p.with_name(p.stem + ".generated.json")


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def _tokens(s: str) -> set[str]:
    return set(_WORD.findall(str(s).lower()))


def _load(path: str | Path) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        return {}


def _covered(*topic_dicts: dict) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """For each source, the set of normalized queries and the set of their word tokens."""
    queries: dict[str, set[str]] = {}
    words: dict[str, set[str]] = {}
    for td in topic_dicts:
        for kind in ("documents", "structured"):
            for source, qs in (td.get(kind) or {}).items():
                for q in qs:
                    if not str(q).strip():
                        continue
                    queries.setdefault(source, set()).add(_norm(q))
                    words.setdefault(source, set()).update(_tokens(q))
    return queries, words


def _is_new(source: str, query: str, *, key: str | None,
            queries: dict[str, set[str]], words: dict[str, set[str]]) -> bool:
    """True if `query` adds coverage the existing watchlist doesn't already have."""
    nq = _norm(query)
    have_q = queries.get(source, set())
    if nq in have_q:
        return False
    # near-duplicate: one query fully contains the other
    if any(nq in h or h in nq for h in have_q):
        return False
    # entity queries (gene/drug): skip if that token is already searched in this source
    if key and key.lower() in words.get(source, set()):
        return False
    return True


# --------------------------------------------------------------------------- #
def _candidates(facts: dict) -> list[dict]:
    """Flatten the facts into ranked (source, query, key, reason, signal) proposals."""
    out: list[dict] = []

    def add(source, query, *, key, reason, signal):
        out.append({"source": source, "query": query, "key": key,
                    "reason": reason, "signal": int(signal)})

    asy = facts.get("asymmetries", {}) or {}
    for row in asy.get("structurally_studied_but_undrugged", []):
        gene, structures, papers = (row + [0, 0])[:3]
        if not gene:
            continue
        add("europepmc", f"{gene} structure function disease mechanism", key=gene,
            reason=f"undrugged target: {structures} structures, {papers} papers", signal=structures + 5)
        add("uniprot", gene, key=gene,
            reason=f"undrugged target with {structures} structures", signal=structures + 5)

    g = facts.get("gaps", {}) or {}
    for row in g.get("targets_without_drugs", []):
        gene = row[0]
        if gene:
            add("europepmc", f"{gene} inhibitor ligand drug discovery", key=gene,
                reason="target with no linked drug", signal=3)
            add("uniprot", gene, key=gene, reason="target with no linked drug", signal=3)
    for row in g.get("drugs_without_papers", []):
        drug = row[0]
        if drug:
            add("europepmc", f"{drug} pharmacology mechanism of action", key=drug,
                reason="drug with no strong paper link", signal=2)
            add("pubchem", drug, key=drug, reason="drug with no strong paper link", signal=2)

    # high-connectivity targets worth deepening (structures present, few papers)
    for row in facts.get("top_targets", []):
        gene, drugs, structures, papers = (list(row) + [0, 0, 0, 0])[:4]
        if gene and structures and papers is not None and papers <= 2:
            add("europepmc", f"{gene} signaling pharmacology therapeutic target", key=gene,
                reason=f"{structures} structures but only {papers} papers", signal=structures)

    t = facts.get("topics", {}) or {}
    for term, n in t.get("mesh", []):
        term = str(term).strip()
        if term and _norm(term) not in _GENERIC and len(term) > 3:
            add("europepmc", term, key=None, reason=f"frequent MeSH term ({n} papers)", signal=n)
    for term, n in t.get("concepts", []):
        term = str(term).strip()
        if term and _norm(term) not in _GENERIC and len(term) > 3:
            add("openalex", term, key=None, reason=f"frequent OpenAlex concept ({n} papers)", signal=n)

    out.sort(key=lambda c: -c["signal"])
    return out


_DOC_SOURCES = {"europepmc", "openalex", "arxiv"}


def generate(con, topics_path: str | Path, *, per_source_cap: int = 3,
             total_cap: int = 40, dry_run: bool = False, now: datetime | None = None) -> list[dict]:
    """Propose + persist new queries from the current corpus. Returns the additions.

    Reads `topics_path` (curated) and its `.generated.json` sibling to know what's
    already covered, mines `analysis.facts(con)` for candidates, and appends the best
    novel ones to the generated file (up to `per_source_cap` per source per run and
    `total_cap` generated queries overall). `dry_run` returns the proposals without
    writing.
    """
    curated = _load(topics_path)
    gen_path = generated_path(topics_path)
    generated = _load(gen_path)
    queries, words = _covered(curated, generated)

    facts = analysis.facts(con)
    existing_generated = sum(len(v) for k in ("documents", "structured")
                             for v in (generated.get(k) or {}).values())
    budget = max(0, total_cap - existing_generated)

    added: list[dict] = []
    per_source: dict[str, int] = {}
    for c in _candidates(facts):
        if len(added) >= budget:
            break
        src = c["source"]
        if per_source.get(src, 0) >= per_source_cap:
            continue
        if not _is_new(src, c["query"], key=c["key"], queries=queries, words=words):
            continue
        added.append(c)
        per_source[src] = per_source.get(src, 0) + 1
        # record so later candidates this run don't duplicate it
        queries.setdefault(src, set()).add(_norm(c["query"]))
        words.setdefault(src, set()).update(_tokens(c["query"]))

    if added and not dry_run:
        _write(gen_path, generated, added, now=now)
    return added


def _write(gen_path: Path, generated: dict, added: list[dict], *, now: datetime | None) -> None:
    stamp = (now or datetime.now(timezone.utc)).date().isoformat()
    generated.setdefault("_generated", True)
    generated.setdefault(
        "_comment",
        "Auto-generated from corpus insights by `aqueduct suggest`. harvest merges "
        "these with topics.json. Reviewable + revertible — delete any entry freely; "
        "it will only come back if the corpus still warrants it.")
    prov = generated.setdefault("_provenance", {})
    for c in added:
        kind = "documents" if c["source"] in _DOC_SOURCES else "structured"
        bucket = generated.setdefault(kind, {}).setdefault(c["source"], [])
        if c["query"] not in bucket:
            bucket.append(c["query"])
        prov[f"{c['source']}\t{c['query']}"] = {
            "reason": c["reason"], "signal": c["signal"], "added": stamp}
    gen_path.parent.mkdir(parents=True, exist_ok=True)
    gen_path.write_text(json.dumps(generated, indent=2, sort_keys=False))
