"""Email delivery via Resend (used to send generated reports).

Credentials come from the environment (never hard-coded):
  RESEND_API_KEY     — Resend API key
  AQUEDUCT_EMAIL_TO  — default recipient
  AQUEDUCT_EMAIL_FROM — optional sender (default Resend's onboarding address)

Put these in an untracked `.secrets.env` and source it from scripts/harvest.sh.
NOTE: Resend sits behind Cloudflare, which 403s requests without a browser-like
User-Agent — so we always send one.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request

RESEND_URL = "https://api.resend.com/emails"
DEFAULT_FROM = "Aqueduct <onboarding@resend.dev>"
_UA = "Mozilla/5.0 (aqueduct/0.1)"


class MailUnavailable(RuntimeError):
    """Raised when no Resend credentials / recipient are configured."""


def available() -> bool:
    return bool(os.environ.get("RESEND_API_KEY"))


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inline(s: str) -> str:
    s = _esc(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def md_to_html(md: str) -> str:
    """Small markdown -> HTML (headers, bold, code, lists, tables, hr, paragraphs)."""
    out, i, lines = [], 0, md.splitlines()
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            i += 1
            continue
        if ln.lstrip().startswith("|") and i + 1 < len(lines) and set(lines[i + 1].replace("|", "").strip()) <= {"-", " ", ":"}:
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], rows[2:]  # rows[1] is the --- separator
            th = "".join(f"<th>{_inline(c)}</th>" for c in head)
            trs = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in body)
            out.append(f'<table border="1" cellpadding="4" cellspacing="0"><tr>{th}</tr>{trs}</table>')
            continue
        m = re.match(r"(#{1,6})\s+(.*)", ln)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
        elif ln.strip() in ("---", "***", "___"):
            out.append("<hr>")
        elif re.match(r"\s*[-*]\s+", ln):
            items = []
            while i < len(lines) and re.match(r"\s*[-*]\s+", lines[i]):
                items.append(f"<li>{_inline(re.sub(r'^\s*[-*]\s+', '', lines[i]))}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        else:
            out.append(f"<p>{_inline(ln)}</p>")
        i += 1
    style = ("<style>body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;"
             "line-height:1.5;color:#222;max-width:760px}table{border-collapse:collapse;"
             "font-size:13px}code{background:#f3f3f3;padding:1px 4px;border-radius:3px}"
             "h1,h2,h3{margin-top:1.2em}</style>")
    return f"<html><head>{style}</head><body>{''.join(out)}</body></html>"


def send(subject: str, markdown: str, to: str | None = None,
         sender: str | None = None) -> str:
    """Send a markdown report as an HTML email. Returns the Resend message id."""
    key = os.environ.get("RESEND_API_KEY")
    if not key:
        raise MailUnavailable("set RESEND_API_KEY to send email")
    to = to or os.environ.get("AQUEDUCT_EMAIL_TO")
    if not to:
        raise MailUnavailable("no recipient (set AQUEDUCT_EMAIL_TO or pass to=)")
    payload = {
        "from": sender or os.environ.get("AQUEDUCT_EMAIL_FROM", DEFAULT_FROM),
        "to": [to], "subject": subject,
        "html": md_to_html(markdown), "text": markdown,
    }
    req = urllib.request.Request(
        RESEND_URL, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r).get("id", "")
