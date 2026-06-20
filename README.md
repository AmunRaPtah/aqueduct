# Aqueduct

A minimal, end-to-end data pipeline: **ingest → store → process → analyze**.

Built on [DuckDB](https://duckdb.org) — a zero-config, in-process analytical database — so the
whole pipeline runs locally with no servers to manage.

```
  sources ──▶ ingest ──▶  raw/        ──▶ store ──▶  bronze ──▶ process ──▶ silver/gold ──▶ analytics
            (JSONL)     (landing zone)            (raw tables)           (clean + modeled)   (queries)
```

## Layout

```
src/aqueduct/
  config.py            paths & settings
  sources/
    europepmc.py       connector: search + full-text JATS-XML fetch
  jats.py              JATS-XML -> metadata + sections (stdlib parser)
  documents.py         doc pipeline: store/process/chunk + report + search
  corpus.py            orchestrate the full-text corpus pipeline
  cli.py               command-line entrypoint
  ingest.py storage.py process.py analytics.py pipeline.py   (synthetic-events demo)
data/                  landing zone + DuckDB warehouse (gitignored)
```

## Real corpus: scientific / medical full text (Europe PMC)

The `corpus` pipeline ingests **full-text** open-access articles — not just metadata.

- **Discovery** → Europe PMC REST search (rich query syntax, keyless)
- **Full text** → NCBI E-utilities `efetch` JATS-XML (Europe PMC's own XML route is unreliable; efetch is not). Keyless; set `NCBI_EMAIL` to identify your traffic and stay under NCBI's 3 req/s.

```bash
export NCBI_EMAIL="you@example.com"
.venv/bin/python -m aqueduct corpus run --source europepmc --query "CRISPR base editing therapy" --limit 25
.venv/bin/python -m aqueduct corpus run --source arxiv     --query "mathematical modeling addiction" --limit 25
.venv/bin/python -m aqueduct corpus search "off-target" -k 5
```

Sources land into the **same** `documents_raw` table (store aggregates every
source's manifest); parsing is dispatched per source (`europepmc`→JATS, `arxiv`→Atom).

**Incremental:** every fetch *accumulates* — batches merge into the landing zone
de-duplicated by primary key (documents by PMCID/arXiv id, drugs by ChEMBL id, etc.),
so repeated or varied queries grow the corpus instead of overwriting it. Re-fetching the
same query is idempotent.

| Source | Fields | Full text? |
|--------|--------|-----------|
| `europepmc` | biomedical: pharma, neuro, pharmacology, epigenetics, genomics | ✅ JATS body + rich metadata (MeSH, keywords, grants, citations, abstract) |
| `arxiv` | math, computational/mathematical modeling, emerging tech, first principles | metadata + abstract; **full PDF body** with `--fulltext` (needs `pip install -e ".[pdf]"`) |
| `openalex` | **every discipline** (physics, CS, math, social science) + all preprint servers | metadata + abstract (keyless) |
| `patents` | USPTO patents — chemistry, devices, ML, materials | title + abstract; needs free `PATENTSVIEW_API_KEY` |

## Structured datasets (records, not documents)

Drug-discovery / clinical / protein data are structured records, not articles. The
`data` pipeline lands flat JSONL and loads it into typed DuckDB tables (separate
tables per source; cross-source links are a later step).

```bash
.venv/bin/python -m aqueduct data fetch --source chembl --query "opioid" --limit 100
.venv/bin/python -m aqueduct data build      # JSONL -> chembl_molecules (typed)
.venv/bin/python -m aqueduct data report
```

```bash
.venv/bin/python -m aqueduct data fetch --source clinicaltrials --query "opioid use disorder" --limit 100
.venv/bin/python -m aqueduct data build      # loads every structured source present
.venv/bin/python -m aqueduct data report
```

| Source | Domain | Tables |
|--------|--------|--------|
| `chembl` | compounds / drug discovery | `chembl_molecules`, `chembl_synonyms`, `chembl_mechanisms` |
| `clinicaltrials` | interventions / therapeutics | `clinical_trials` |
| `uniprot` | proteins / drug targets | `uniprot_proteins` |
| `pdb` | protein structures (enriches UniProt refs) | `pdb_structures` |
| `pubchem` | compounds / cheminformatics | `pubchem_compounds` |
| `ensembl` | genomics (gene location/biotype) | `ensembl_genes` |

`data fetch --source ensembl` with no `--query` enriches the genes already in the UniProt
landing zone. Graph links: `link_drug_pubchem` (ChEMBL ↔ PubChem by InChIKey) and
`link_protein_gene` (protein ↔ Ensembl gene by symbol).

`data fetch --source pdb` ignores `--query` — it enriches the structures referenced by
whatever UniProt proteins are already in the landing zone.

## Cross-source links (drug → trial → paper)

`links` connects the silos into a graph with the **drug as hub** — matching a
normalised drug name (salt forms stripped) against trial interventions and document
**full text** + metadata, word-boundary safe.

```bash
.venv/bin/python -m aqueduct links build                  # entity_drugs + link_* tables
.venv/bin/python -m aqueduct links report                 # best-connected drugs
.venv/bin/python -m aqueduct links explore buprenorphine  # its trials + papers
```

| Table | Edge |
|-------|------|
| `entity_drugs` | canonical drug (salt forms merged) + form count + max phase |
| `entity_drug_names` | canonical drug → match terms (base name + synonyms/trade names) |
| `link_drug_trial` | drug ↔ trial (`in_intervention` flag) |
| `link_drug_document` | drug ↔ paper (`in_metadata`, `n_body`, `confidence`) |
| `link_trial_document` | trial ↔ paper, via shared drug (intervention + strong only) |
| `entity_proteins` | canonical protein (UniProt accession, gene) |
| `link_drug_protein` | drug ↔ target protein, via ChEMBL mechanism ↔ UniProt xref |
| `link_protein_structure` | protein ↔ PDB structure (UniProt cross-refs) |
| `entity_protein_names` | protein → match terms (gene symbol + name aliases) |
| `link_protein_document` | protein ↔ paper, with confidence |

Protein↔document matching uses the **gene symbol plus name aliases** (UniProt
recommended/alternative names + short names + gene synonyms). Multiword aliases match
with flexible separators, so a paper saying "mu-opioid receptor" links to `OPRM1` even
when it never writes the gene symbol.

The protein arm makes the graph biological: **drug → target protein → structure**, and
drug → trial → paper. `links protein OPRM1` shows a target's drugs, structures, papers.

```bash
.venv/bin/python -m aqueduct links explore fentanyl   # ... + Target proteins: OPRM1 (AGONIST)
.venv/bin/python -m aqueduct links protein OPRM1       # drugs hitting it + PDB structures
```

### Semantic-over-graph discovery
`links discover` joins the two big subsystems: it builds a **concept profile** for a drug
from the graph (target protein names + function, mechanism of action, trial conditions),
then runs that profile through semantic search. Papers conceptually about the drug's
biology surface **even when they never name the drug**, each tagged DIRECT (already a
lexical link) or SEMANTIC (newly discovered). Requires `links build` + `corpus index`.

```bash
.venv/bin/python -m aqueduct links discover buprenorphine -k 6
#   targets: Mu-type opioid receptor | mechanism: Mu/Kappa opioid receptor agonist
#   -> e.g. "Opioid receptor distribution in the claustrum" (SEMANTIC — drug unnamed)
```

**Entity resolution:** canonical drug = normalised base name; match terms expand with
ChEMBL synonyms/trade names (so "Sublocade"/"Subutex" → buprenorphine). Drug↔document
links carry a **confidence**: a hit in title/abstract/keywords/MeSH or ≥2 full-text
mentions is `strong`; a single body co-mention is `weak` (hidden by default in
`report`/`explore`). Still lexical — id-level resolution (ChEMBL/MeSH/RxNorm) is the
next refinement.

Layers (medallion / ELT — land raw, transform in-warehouse so you can reprocess
without re-fetching):

| Stage   | Input                        | Output          | Layer  |
|---------|------------------------------|-----------------|--------|
| fetch   | Europe PMC query             | `raw/europepmc/*.xml` + manifest | landing |
| store   | landing XML + manifest       | `documents_raw` (raw JATS kept)  | bronze |
| process | `documents_raw`              | `doc_sections` (title/abstract/body, heading paths) | silver |
| chunk   | `doc_sections`               | `doc_chunks` (≤220-word windows, search/embedding-ready) | gold |

`corpus run` = fetch → store → process → chunk → report. Sub-commands
(`fetch`/`build`/`report`/`search`) run stages individually.

### Semantic search
Concept-level search over chunks via a pluggable embedder — default **LSA**
(TF-IDF + truncated SVD, pure NumPy: no API key, no model download). Build the index
after (re)building the corpus, then query by meaning rather than keywords:

```bash
.venv/bin/python -m aqueduct corpus index                              # embed all chunks
.venv/bin/python -m aqueduct corpus semantic "reversing opioid overdose" -k 5
```

The index is a derived sidecar (`data/lsa_model.json` + `data/chunk_index.npz`) and records
which **backend** produced it, so search reconstructs the right embedder.

**Backends** (registry in `embeddings.BACKENDS`):

| `--backend` | What | Deps |
|-------------|------|------|
| `auto` (default) | st if installed, else lsa | — |
| `lsa` | TF-IDF + truncated SVD | keyless, NumPy only |
| `st` | sentence-transformers (e.g. `all-MiniLM-L6-v2`) | `pip install -e ".[st]"` |

```bash
.venv/bin/python -m aqueduct corpus index                      # auto-selects best available
.venv/bin/python -m aqueduct corpus index --backend lsa        # force keyless LSA
```

The loaded transformer model is cached process-wide, so library use that both indexes and
queries doesn't pay the load twice.

Add a backend by subclassing `Embedder` (implement `fit`/`transform`/`state`/`from_state`)
and registering it in `BACKENDS` — storage, search, and discovery are backend-agnostic.

### Adding sources
Drop a new connector in `src/aqueduct/sources/` that lands `<id>.xml` files plus a
`manifest.jsonl` under `data/raw/<source>/`. `store_documents` picks up every
source's manifest automatically — patents (PatentsView), preprints, etc.

## Tests

Deterministic and **offline** — no network. Synthetic fixtures are seeded into a temp
landing zone + DuckDB warehouse, then asserted through the real pipeline functions:
parsers, type coercion, entity normalization, the document pipeline, dataset loading,
and the full link graph (drug ⇄ trial ⇄ paper ⇄ protein ⇄ structure).

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q          # ~28 tests, a few seconds
```

**CI:** `.github/workflows/ci.yml` runs the suite on Python 3.10–3.12 on every push /
PR (activates once the repo is pushed to GitHub). Tests are offline, so CI needs no
network or secrets.

## Demo: synthetic events pipeline

The original scaffold, kept as a runnable reference for the same ELT shape:

```bash
.venv/bin/python -m aqueduct run --events 5000   # ingest -> store -> process -> analytics
.venv/bin/python -m aqueduct query               # analytics summary
```

| Stage      | Input                | Output                 | Layer        |
|------------|----------------------|------------------------|--------------|
| ingest     | source (synthetic)   | `data/raw/*.jsonl`     | landing      |
| store      | `data/raw/*.jsonl`   | `events_raw`           | bronze       |
| process    | `events_raw`         | `events`, `daily_*`    | silver/gold  |
| analytics  | modeled tables       | printed report         | —            |
