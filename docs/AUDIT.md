# Aqueduct audit — findings & status

External critical audit (2026-06-20), with the lens of using Aqueduct as a **RAG
backend for an external agent** (the Pardalos project). Verdict at audit time:
🟢 local research use ready · 🔴 external-RAG use *not* ready (5 blockers).

Status legend: ✅ done · 🟡 partial · ⬜ open

## What it does well
Clean medallion architecture; keyless idempotent connectors; robust JATS parser;
pluggable embeddings with incremental re-embed; lexical drug normalization + ChEMBL
synonym graph; declarative topics-driven harvest that survives per-query failures.

## The 5 RAG blockers
| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Retrieval/query API | 🟡 | `aqueduct rag` returns structured JSON (chunks+citations+graph) + filters. **REST/HTTP endpoint still needed for a *remote* consumer.** |
| 2 | Chunk-level metadata | 🟡 | `sec_type`/`sec_title` exist + exposed; finer flags (is_methods/results, figure/table counts, ordinal) open. |
| 3 | Index versioning/validation | ✅ | Index now stamps `index_version/backend/dims/n_chunks/corpus_hash`; `index_info()` exposes it. |
| 4 | Idempotency / orphaned embeddings | 🟡 | Incremental embedding reuses unchanged chunks by hash; a `corpus_state` hash-check to auto-skip/rebuild is open. |
| 5 | Retrieval-quality benchmark | ✅ | `tests/test_retrieval_quality.py` — Recall@k + MRR gold set, run on backend/chunking changes. |

## Other findings (prioritized backlog)
| Pri | Item | Status |
|-----|------|--------|
| MED | Structured error handling (Transient vs Permanent) | ⬜ |
| MED | Shared RateLimiter + 429/header honoring | ⬜ |
| MED | Exp. backoff w/ jitter + circuit breaker | ⬜ |
| MED | Retrieval filtering (date/source/section) + min-score | ✅ (`rag --min-score/--source/--section`) |
| LOW | arXiv PDF full-text cache (avoid re-download) | ⬜ |
| — | Sentence/section-boundary-aware + configurable chunking | ⬜ |
| — | Numeric (0–1) drug↔document confidence scoring | ⬜ |
| — | Drug↔protein matching: restrict to sentence/noun-phrase (fewer false positives) | ⬜ |
| — | discover.py: weight profile components (targets ×3, names ×2) | ⬜ |
| — | Entity resolution: RxNorm/MeSH harmonization, stereoisomers | ⬜ |
| — | Validation phase (text length, word-count, id format) | ⬜ |
| — | Structured JSON logging / observability | ⬜ |
| — | Harvest query-version tracking (refresh stale results) | ⬜ |

## Closed this cycle
✅ #3 index versioning · ✅ #5 quality benchmark · ✅ retrieval filtering/min-score ·
🟡 #1 retrieval API (JSON CLI) · 🟡 #2 chunk metadata exposed · 🟡 #4 incremental embedding.
