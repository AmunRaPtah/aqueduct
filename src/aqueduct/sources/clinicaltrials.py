"""ClinicalTrials.gov connector (therapeutics / interventions — structured data).

Uses the ClinicalTrials.gov REST API v2 (keyless). Lands flat trial records as
JSONL in the structured landing zone, for the `data` (structured-mode) pipeline.
"""

from __future__ import annotations

import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

from .. import config, net
from ..landing import merge_jsonl

API = "https://clinicaltrials.gov/api/v2/studies"
USER_AGENT = "aqueduct/0.1 (data pipeline)"
PAGE_DELAY = 0.2


def _get(url: str, *, retries: int = 3, timeout: int = 30) -> dict:
    """Fetch JSON via the shared resilient client (retry/backoff/rate-limit/breaker)."""
    return net.get_json(url, timeout=timeout, retries=retries)


def _flatten(study: dict) -> dict:
    ps = study.get("protocolSection", {})
    idm = ps.get("identificationModule", {})
    st = ps.get("statusModule", {})
    dz = ps.get("designModule", {})
    enroll = dz.get("enrollmentInfo", {})
    conds = ps.get("conditionsModule", {}).get("conditions", [])
    ivs = ps.get("armsInterventionsModule", {}).get("interventions", [])
    spon = ps.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {})
    cnt = enroll.get("count")
    return {
        "nct_id": idm.get("nctId"),
        "title": idm.get("briefTitle"),
        "status": st.get("overallStatus"),
        "study_type": dz.get("studyType"),
        "phases": "; ".join(dz.get("phases", []) or []) or None,
        "enrollment": int(cnt) if isinstance(cnt, (int, float)) else None,
        "start_date": st.get("startDateStruct", {}).get("date"),
        "completion_date": st.get("completionDateStruct", {}).get("date"),
        "conditions": "; ".join(conds) or None,
        "interventions": "; ".join(
            f"{i.get('type')}:{i.get('name')}" for i in ivs if i.get("name")
        ) or None,
        "lead_sponsor": spon.get("name"),
    }


#: which ClinicalTrials.gov v2 search field a harvest query targets.
QUERY_FIELDS = {"cond": "query.cond", "lead": "query.lead"}


def search(query: str, limit: int = 100,
           cursor: str | None = None, field: str = "cond") -> tuple[list[dict], str | None]:
    """Search trials by condition/term (`field="cond"`) or lead sponsor (`field="lead"`).

    Returns ``(records, next_cursor)``: flattened records starting from `cursor` (the
    `pageToken` a previous run left off at), plus the token to resume from next run —
    so successive harvests page deeper instead of re-reading the first page.
    `next_cursor` is None at end-of-results (caller resweeps from the top next cycle).
    """
    out: list[dict] = []
    token: str | None = cursor or None
    next_cursor: str | None = None
    param = QUERY_FIELDS[field]
    while len(out) < limit:
        page = min(200, limit - len(out))
        params = {param: query, "pageSize": page, "countTotal": "false"}
        if token:
            params["pageToken"] = token
        data = _get(f"{API}?{urllib.parse.urlencode(params)}")
        studies = data.get("studies", [])
        if not studies:
            next_cursor = None  # exhausted -> resweep next cycle
            break
        out.extend(_flatten(s) for s in studies)
        token = data.get("nextPageToken")
        next_cursor = token
        if not token:
            break
        time.sleep(PAGE_DELAY)
    return out[:limit], next_cursor


def ingest(query: str, limit: int = 100,
           cursor: str | None = None, field: str = "cond") -> tuple[Path, str | None]:
    """Land ClinicalTrials.gov studies as JSONL in the structured landing zone.

    Resumes paging from `cursor` and returns ``(landing_file, next_cursor)``.
    """
    src_dir = config.raw_source_dir("clinicaltrials")
    records, next_cursor = search(query, limit=limit, cursor=cursor, field=field)
    out = src_dir / "trials.jsonl"
    fetched_at = datetime.now(timezone.utc).isoformat()
    recs = [{**r, "query": query, "fetched_at": fetched_at} for r in records]
    total, added = merge_jsonl(out, recs, "nct_id")
    print(f"[ingest]  clinicaltrials ({field}): +{added} new trials ({total} total) for {query!r} -> {out.relative_to(config.ROOT)}")
    return out, next_cursor


def ingest_sponsor(query: str, limit: int = 100,
                    cursor: str | None = None) -> tuple[Path, str | None]:
    """Land trials led by `query` (a sponsor/company name) — see `ingest`.

    Registered as the `clinicaltrials_sponsor` harvest source so `topics.json` can
    seed coverage by company name, not just disease/condition term. Lands into the
    same `trials.jsonl` (deduped by `nct_id`) as condition-based harvests.
    """
    return ingest(query, limit=limit, cursor=cursor, field="lead")
