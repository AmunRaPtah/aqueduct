"""ClinicalTrials.gov connector (therapeutics / interventions — structured data).

Uses the ClinicalTrials.gov REST API v2 (keyless). Lands flat trial records as
JSONL in the structured landing zone, for the `data` (structured-mode) pipeline.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from ..landing import merge_jsonl

API = "https://clinicaltrials.gov/api/v2/studies"
USER_AGENT = "aqueduct/0.1 (data pipeline)"
PAGE_DELAY = 0.2


def _get(url: str, *, retries: int = 3, timeout: int = 30) -> dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url}") from last


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


def search(query: str, limit: int = 100) -> list[dict]:
    """Search trials by condition/term; returns flattened records."""
    out: list[dict] = []
    token: str | None = None
    while len(out) < limit:
        page = min(200, limit - len(out))
        params = {"query.cond": query, "pageSize": page, "countTotal": "false"}
        if token:
            params["pageToken"] = token
        data = _get(f"{API}?{urllib.parse.urlencode(params)}")
        studies = data.get("studies", [])
        if not studies:
            break
        out.extend(_flatten(s) for s in studies)
        token = data.get("nextPageToken")
        if not token:
            break
        time.sleep(PAGE_DELAY)
    return out[:limit]


def ingest(query: str, limit: int = 100) -> Path:
    """Land ClinicalTrials.gov studies as JSONL in the structured landing zone."""
    src_dir = config.raw_source_dir("clinicaltrials")
    records = search(query, limit=limit)
    out = src_dir / "trials.jsonl"
    fetched_at = datetime.now(timezone.utc).isoformat()
    recs = [{**r, "query": query, "fetched_at": fetched_at} for r in records]
    total, added = merge_jsonl(out, recs, "nct_id")
    print(f"[ingest]  clinicaltrials: +{added} new trials ({total} total) for {query!r} -> {out.relative_to(config.ROOT)}")
    return out
