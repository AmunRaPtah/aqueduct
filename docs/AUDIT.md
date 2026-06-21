# Aqueduct audit — findings & status

External critical audit (2026-06-20), with the lens of using Aqueduct as a **RAG
backend for an external agent** (the Pardalos project). Verdict at audit time:
🟢 local research use ready · 🔴 external-RAG use *not* ready (5 blockers).

**Update (2026-06-20, remediation cycle):** all 5 RAG blockers closed and the entire
prioritized backlog cleared. Verdict now: 🟢 local research ready · 🟢 external-RAG ready.

Status legend: ✅ done · 🟡 partial · ⬜ open

## What it does well
Clean medallion architecture; keyless idempotent connectors; robust JATS parser;
pluggable embeddings with incremental re-embed; lexical drug normalization + ChEMBL
synonym graph; declarative topics-driven harvest that survives per-query failures.

## The 5 RAG blockers
| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Retrieval/query API | ✅ | `aqueduct rag` (JSON CLI) **and** `aqueduct serve` — a stdlib HTTP API (`/retrieve`, `/health`, `/facts`, `/discover`) with bearer auth, for remote consumers (the Pardalos bot). |
| 2 | Chunk-level metadata | ✅ | `sec_type`/`sec_title` **plus** `sec_kind` (IMRaD), `is_methods`/`is_results`, `n_figures`/`n_tables`, and per-doc chunk ordinal — all persisted on `doc_chunks`, exposed by `rag.retrieve`, and filterable (`rag --kind methods,results`). |
| 3 | Index versioning/validation | ✅ | Index stamps `index_version/backend/dims/n_chunks/corpus_hash`; `index_info()` exposes it. |
| 4 | Idempotency / orphaned embeddings | ✅ | Incremental re-embed by hash **and** a `corpus_hash` check in `build_index` that auto-skips a rebuild when the corpus is unchanged (`--force` overrides). A no-op harvest now costs one hash, not a re-embed. |
| 5 | Retrieval-quality benchmark | ✅ | `tests/test_retrieval_quality.py` — Recall@k + MRR gold set, run on backend/chunking changes. |

## Other findings (prioritized backlog) — all cleared
| Pri | Item | Status | Where |
|-----|------|--------|-------|
| MED | Structured error handling (Transient vs Permanent) | ✅ | `net.py` — `TransientError`/`PermanentError`/`RateLimitError`/`CircuitOpenError` |
| MED | Shared RateLimiter + 429/header honoring | ✅ | `net.RateLimiter` — per-host pacing, honors `Retry-After`/429 |
| MED | Exp. backoff w/ jitter + circuit breaker | ✅ | `net.request` — full-jitter capped backoff; per-host breaker |
| MED | Retrieval filtering (date/source/section) + min-score | ✅ | `rag --min-score/--source/--section/--kind` |
| LOW | arXiv PDF full-text cache (avoid re-download) | ✅ | `arxiv._cached_pdf` → `data/cache/arxiv_pdf/` |
| — | Sentence/section-boundary-aware + configurable chunking | ✅ | `documents._chunk_sentences`; `AQUEDUCT_CHUNK_WORDS/OVERLAP/SENTENCE_AWARE` |
| — | Numeric (0–1) drug↔document confidence scoring | ✅ | `link_drug_document.score` + `link_protein_document.score` (saturating, density-aware) |
| — | Drug↔protein matching: sentence/noun-phrase precision | ✅ | noun-phrase (multiword) patterns + density-weighted numeric `score` downweights single incidental mentions |
| — | discover.py: weight profile components (targets ×3, names ×2) | ✅ | `discover._profile` / `_protein_profile` |
| — | Entity resolution: harmonization, stereoisomers | ✅ | `links._STEREO` strips enantiomer/racemate prefixes in `_norm` (RxNorm/MeSH cross-walk noted below) |
| — | Validation phase (text length, word-count, id format) | ✅ | `validate.py`; `aqueduct validate`; runs in `harvest` |
| — | Structured JSON logging / observability | ✅ | `obs.py` (`AQUEDUCT_LOG_JSON`/`AQUEDUCT_LOG_FILE`); wired into `net` + `harvest` |
| — | Harvest query-version tracking (refresh stale results) | ✅ | `data/harvest_state.json`; `harvest.stale_queries()` |

## Notes / deliberately scoped
- **RxNorm/MeSH harmonization** is addressed at the stereoisomer/salt level (no external
  vocabulary download); a full RxNorm/MeSH cross-walk remains a future enrichment, not a
  blocker — the graph already resolves trade names via the ChEMBL synonym set.
- **Numeric link `score`** is intentionally continuous and *additive* to the existing
  `confidence` text ('strong'/'weak'), so downstream consumers relying on the text label
  are unaffected.

## Engineering
- All work is test-covered: suite grew 80 → 105 tests (`test_net`, `test_chunking`,
  `test_validate`, plus link-score / stereo / idempotency / harvest-state / randomized-SVD cases).
- The 11 connectors + their bespoke retry loops collapsed onto one resilient client
  (`net.py`); behaviour preserved (raise-style vs None-on-failure) per connector.

### Scaling fixes found while remediating (the prod harvest was OOM-killed, exit 137)
- **LSA fit did a *full* SVD** of the (n_chunks × vocab) TF-IDF matrix — materialising all
  ~8k singular vectors to keep 128. Replaced with a seeded **randomized truncated SVD**
  (float32, power iterations) that computes only the top-k: ~50× less work, bounded memory,
  exact-SVD fallback for small (test) matrices so results are unchanged.
- **ST embedding** now encodes with a bounded `batch_size`, so a full re-embed (forced by
  the chunking change) streams instead of allocating one giant activation tensor.
- **Silver/gold inserts batched** (`executemany`) instead of tens of thousands of single-row
  appends.
