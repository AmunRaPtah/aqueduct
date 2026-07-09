"""Ingest-time quality gate — admission control for the corpus.

`validate.py` *reports* garbage after it is already stored; this module stops it at
the door. Every candidate document is checked *before* it lands in `documents_raw`:
accepted rows enter the corpus, rejected rows are quarantined in `documents_rejected`
with the failing reason(s) — never silently dropped, so a too-strict rule stays
recoverable and every rejection is auditable.

A document failing ANY hard check is quarantined:
  missing_title          no usable title
  title_too_short        title below the minimum length (junk / nav cruft)
  bad_id                 empty document id
  year_out_of_range      pub_year outside sane bounds (parse error / bad metadata)
  no_indexable_content   neither a real body nor a usable abstract — nothing to retrieve
  body_missing           has_body claimed but the parsed body is empty/truncated (broken fetch)
  withdrawn_or_retracted the record is a retraction/withdrawal notice, not knowledge
  garbage_text           text is mostly non-letters or pathologically repetitive (OCR/encoding junk)

Thresholds are read from the environment on every call (AQUEDUCT_Q_*), so a deploy —
or a test — can loosen/tighten them without code changes or import-order games.
Defaults are production-conservative; set the length mins to 0 to disable them.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field

# Production defaults. Overridden per-call by AQUEDUCT_Q_<NAME> if set.
_DEFAULTS = {
    "MIN_TITLE_CHARS": 8,
    "MIN_BODY_WORDS": 40,
    "MIN_ABSTRACT_WORDS": 10,
    "YEAR_MIN": 1500,
    "YEAR_MAX": 2100,
    "MIN_ALPHA_RATIO": 0.5,
    "MAX_REPEAT_RATIO": 0.6,
}


def _t(name: str) -> float:
    """Current threshold for `name` — env override (AQUEDUCT_Q_<name>) or default."""
    raw = os.environ.get(f"AQUEDUCT_Q_{name}")
    if raw is None:
        return _DEFAULTS[name]
    try:
        return float(raw)
    except ValueError:
        return _DEFAULTS[name]


# Match retraction/withdrawal notices *specifically* — as a title prefix or a stock
# abstract sentence — so a paper merely discussing retractions isn't falsely rejected.
_RETRACTION_TITLE = re.compile(
    r"^\s*(retracted|retraction|withdrawn|withdrawal|"
    r"(editorial )?expression of concern)\b[:\s.-]", re.I)
_RETRACTION_ABSTRACT = re.compile(
    r"this article has been (withdrawn|retracted|removed)", re.I)


@dataclass
class Verdict:
    """Outcome of the gate for one document. `ok` False => quarantine."""
    ok: bool
    reasons: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)


def _alpha_ratio(text: str) -> float:
    """Fraction of non-space characters that are letters. Low => OCR/encoding junk."""
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return 1.0  # emptiness is handled by the content checks, not flagged as garbage
    return sum(c.isalpha() for c in non_space) / len(non_space)


def _repeat_ratio(text: str) -> float:
    """Share of tokens held by the single most common token. High => degenerate text."""
    toks = text.split()
    if len(toks) < 20:
        return 0.0  # too short to judge repetition reliably
    return Counter(toks).most_common(1)[0][1] / len(toks)


def check_document(rec: dict, *, body_words: int | None = None) -> Verdict:
    """Admit or quarantine one candidate document.

    `body_words` is the parsed body length; pass it so a `has_body` claim can be
    verified against real content. None means "not measured" — the body-length check
    is then skipped rather than assumed to fail.
    """
    reasons: list[str] = []
    title = (rec.get("title") or "").strip()
    abstract = (rec.get("abstract") or "").strip()
    doc_id = (rec.get("pmcid") or "").strip()
    has_body = bool(rec.get("has_body"))
    year = rec.get("pub_year")
    abstract_words = len(abstract.split())

    if not title:
        reasons.append("missing_title")
    elif len(title) < _t("MIN_TITLE_CHARS"):
        reasons.append("title_too_short")

    if not doc_id:
        reasons.append("bad_id")

    if year is not None:
        try:
            y = int(year)
            if y < _t("YEAR_MIN") or y > _t("YEAR_MAX"):
                reasons.append("year_out_of_range")
        except (TypeError, ValueError):
            reasons.append("year_out_of_range")

    if _RETRACTION_TITLE.match(title) or _RETRACTION_ABSTRACT.search(abstract[:300]):
        reasons.append("withdrawn_or_retracted")

    # A document must carry something retrievable: a real body OR a usable abstract.
    # An empty body (0 words) while has_body is claimed is always a broken fetch,
    # independent of the (tunable) length threshold.
    min_body = _t("MIN_BODY_WORDS")
    body_broken = has_body and body_words is not None and (
        body_words == 0 or body_words < min_body)
    body_ok = has_body and not body_broken
    abstract_ok = abstract_words >= _t("MIN_ABSTRACT_WORDS")
    if body_broken:
        reasons.append("body_missing")
    if not body_ok and not abstract_ok and "body_missing" not in reasons:
        reasons.append("no_indexable_content")

    # Garbage detection on whatever human text we have (title + abstract).
    sample = " ".join(t for t in (title, abstract) if t)
    if sample and (_alpha_ratio(sample) < _t("MIN_ALPHA_RATIO")
                   or _repeat_ratio(sample) > _t("MAX_REPEAT_RATIO")):
        reasons.append("garbage_text")

    checks = {
        "title_chars": len(title),
        "abstract_words": abstract_words,
        "body_words": body_words,
        "has_body": has_body,
        "year": year,
    }
    return Verdict(ok=not reasons, reasons=reasons, checks=checks)
