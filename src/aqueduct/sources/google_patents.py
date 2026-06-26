"""Google Patents connector — BigQuery public dataset (`patents-public-data`).

Queries `patents-public-data.patents.publications` via the `bq` CLI. Needs the Google
Cloud SDK installed and authenticated, with a billing/quota project set
(GOOGLE_CLOUD_PROJECT, or `gcloud config set project`). BigQuery's free tier covers
1 TB of query/month. Without `bq` on PATH the connector no-ops with a clear message.

NOTE: credential/tooling-gated and not verified live from this environment (no `bq`
installed here). The SQL targets the documented publications schema (localized title/
abstract arrays); verify once the Cloud SDK is configured.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .. import config
from ..landing import merge_jsonl

TABLE = "patents-public-data.patents.publications"


def _project() -> str | None:
    return os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")


def _available() -> bool:
    return shutil.which("bq") is not None


def search(query: str, limit: int = 25) -> list[dict]:
    """Title/abstract keyword search over Google Patents via `bq`. Needs the Cloud SDK."""
    if not _available():
        return []
    kw = query.lower().replace("\\", "").replace("'", "").replace("%", "")
    sql = f"""
      SELECT publication_number AS patent_number,
             country_code AS country,
             (SELECT t.text FROM UNNEST(title_localized) t
                WHERE t.language = 'en' LIMIT 1) AS title,
             (SELECT a.text FROM UNNEST(abstract_localized) a
                WHERE a.language = 'en' LIMIT 1) AS abstract,
             CAST(publication_date AS STRING) AS pub_date,
             (SELECT x.name FROM UNNEST(assignee_harmonized) x LIMIT 1) AS assignee
      FROM `{TABLE}`
      WHERE EXISTS (SELECT 1 FROM UNNEST(abstract_localized) a
                    WHERE a.language = 'en' AND LOWER(a.text) LIKE '%{kw}%')
      LIMIT {int(limit)}
    """
    cmd = ["bq", "query", "--use_legacy_sql=false", "--format=json",
           f"--max_rows={int(limit)}"]
    proj = _project()
    if proj:
        cmd.append(f"--project_id={proj}")
    cmd.append(sql)
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (subprocess.TimeoutExpired, OSError):
        return []
    if out.returncode != 0:
        print(f"[ingest]  google_patents: bq error: {out.stderr.strip()[:200]}")
        return []
    try:
        return json.loads(out.stdout or "[]")
    except json.JSONDecodeError:
        return []


def parse(raw: str) -> dict:
    """Parse a stored Google Patents record (JSON) into {'meta', 'sections'}."""
    try:
        r = json.loads(raw)
    except json.JSONDecodeError:
        return {"meta": {}, "sections": []}
    sections = []
    if r.get("title"):
        sections.append({"sec_type": "title", "sec_title": None, "text": r["title"]})
    if r.get("abstract"):
        sections.append({"sec_type": "abstract", "sec_title": "abstract", "text": r["abstract"]})
    return {"meta": r, "sections": sections}


def ingest(query: str, limit: int = 25) -> Path:
    """Land Google Patents hits for `query`. Returns the landing dir."""
    src_dir = config.raw_source_dir("google_patents")
    manifest = src_dir / "manifest.jsonl"
    if not _available():
        print("[ingest]  google_patents: `bq` (Google Cloud SDK) not found. "
              "Install + authenticate it to enable. Skipping.")
        return src_dir
    records = search(query, limit=limit)
    print(f"[ingest]  google_patents: {len(records)} patents for {query!r}")

    fetched_at = datetime.now(timezone.utc).isoformat()
    built = []
    for r in records:
        pid = r.get("patent_number")
        if not pid:
            continue
        pub = r.get("pub_date") or ""
        json_path = src_dir / f"GP_{pid}.json"
        json_path.write_text(json.dumps(r), encoding="utf-8")
        built.append({
            "pmcid": f"patent:{pid}", "pmid": None, "doi": None,
            "title": r.get("title"), "journal": r.get("assignee"),
            "pub_year": pub[:4] or None, "authors": r.get("assignee"),
            "source": "google_patents", "query": query, "fetched_at": fetched_at,
            "xml_file": config.rel_data_path(json_path), "has_body": False,
            "abstract": r.get("abstract"), "mesh": None, "keywords": None,
            "grants": None, "cited_by": None,
        })
    total, added = merge_jsonl(manifest, built, "pmcid")
    print(f"[ingest]  manifest +{added} new ({total} total) -> {manifest.relative_to(config.ROOT)}")
    return src_dir
