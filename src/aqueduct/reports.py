"""LLM-augmented reporting — DuckDB facts + DeepSeek interpretation, grounded.

Two modes, both fed the zero-token facts sheet so the model spends budget only on
reasoning:

  default : ONE DeepSeek call interprets facts + retrieved excerpts -> report (cheap).
  agent   : a bounded Claude Code harness on the DeepSeek backend digs deeper, with
            read-only query access and capped turns (opt-in; more tokens).

If no LLM credentials are present, a facts-only report is still produced (no tokens).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import analysis, config, embeddings, llm, mailer
from .storage import connect

SYSTEM = (
    "You are a research strategist and innovation analyst — NOT a summarizer. You receive "
    "pre-computed quantitative facts (counts, links, asymmetries, gaps) about a knowledge "
    "base plus a few evidence excerpts.\n\n"
    "HARD RULES:\n"
    "1. Do NOT summarize individual papers. Do NOT restate textbook or well-known facts. "
    "Do NOT include a 'key themes in the literature' or paper-by-paper section.\n"
    "2. EVERY point must be a NOVEL insight that no single source states: a non-obvious "
    "connection across drugs/targets/trials/genes, a whitespace gap and why it is a "
    "high-value opportunity, a specific testable hypothesis, or a contrarian take on where "
    "the field's attention is misallocated.\n"
    "3. Mine the ASYMMETRIES and GAPS hardest (e.g. a target with many structures but no "
    "drug = druggable whitespace; a drug spanning many conditions = repurposing signal; "
    "research-rich but untrialed drugs = translation gap). Reason about WHY and WHAT TO DO.\n"
    "4. Be specific and grounded in the actual numbers/links; NEVER invent entities or "
    "figures. If the data is thin, say so plainly rather than padding.\n"
    "Output a sharp strategy memo in markdown — dense, non-obvious, actionable."
)


def _snippets(topic: str, k: int = 8, con=None) -> tuple[str, list[tuple]]:
    """Top semantically-relevant chunks for grounding + their citations."""
    hits = embeddings.rank(topic, k=k) if topic else []
    if not hits:
        return "", []
    lines, cites = [], []
    for pmcid, cid, score in hits:
        row = con.execute(
            "SELECT d.title, c.text FROM doc_chunks c JOIN documents_raw d USING (pmcid) "
            "WHERE c.pmcid=? AND c.chunk_id=?", [pmcid, cid]).fetchone()
        if row:
            title, text = row
            lines.append(f"[{pmcid}] {text[:320].strip()}")
            cites.append((pmcid, title))
    return "\n\n".join(lines), cites


def _prompt(sheet: str, snippets: str, topic: str | None) -> str:
    focus = f"Focus: {topic}.\n\n" if topic else "Scope: the whole knowledge base.\n\n"
    body = f"{focus}## Quantitative facts (use these numbers; do not invent)\n{sheet}\n"
    if snippets:
        body += ("\n## Evidence excerpts (use ONLY as supporting evidence for a specific "
                 "insight — do NOT summarize them; cite by [id])\n" + snippets + "\n")
    body += (
        "\nWrite the strategy memo with EXACTLY these sections:\n"
        "0. **Executive summary** — FIRST, 4-5 dense bullets, each ONE bold, punchy novel "
        "insight or opportunity (the highest-value takeaways), readable in 20 seconds, no "
        "fluff. This is the part that matters most; make every bullet count.\n"
        "1. **Novel insights** — 4-6 non-obvious findings from connecting the data. For "
        "each: the insight in one bold line, then the supporting numbers/links, then why "
        "it is not obvious.\n"
        "2. **Druggable & translational whitespace** — the strongest gaps/asymmetries "
        "(undrugged-but-structurally-studied targets, research-rich-but-untrialed drugs), "
        "and why each is a concrete opportunity.\n"
        "3. **Testable hypotheses** — 3-5 specific, falsifiable predictions worth pursuing.\n"
        "4. **Contrarian angle** — where the field's attention looks misallocated, per the "
        "numbers.\n"
        "5. **Highest-value open questions.**\n"
        "No literature summary. No background. Lead with the most surprising thing.")
    return body


def _deepseek_env() -> dict | None:
    cfg = llm.config()
    if not cfg:
        return None
    return {**os.environ, "ANTHROPIC_BASE_URL": cfg["base"],
            "ANTHROPIC_API_KEY": cfg["key"]}


def _agentic(sheet: str, snippets: str, topic: str | None, max_turns: int = 12) -> str:
    """Run a bounded Claude Code agent on DeepSeek to analyse, with read-only DB access."""
    env = _deepseek_env()
    if env is None:
        raise llm.LLMUnavailable("agent mode needs DeepSeek credentials")
    cfg = llm.config()
    sandbox = Path(tempfile.mkdtemp(prefix="aqx-"))
    (sandbox / "facts.md").write_text(sheet)
    if snippets:
        (sandbox / "excerpts.md").write_text(snippets)
    # read-only query helper the agent may call
    qpy = sandbox / "q.py"
    qpy.write_text(
        "import sys, duckdb\n"
        f"con = duckdb.connect({str(config.WAREHOUSE)!r}, read_only=True)\n"
        "print(con.sql(sys.stdin.read()))\n")
    task = (
        f"You are analysing the Aqueduct knowledge base. Pre-computed facts are in "
        f"facts.md (read it first). Focus: {topic or 'overall landscape'}. You may run "
        f"read-only SQL with: echo \"<SELECT ...>\" | python {qpy.name}  (tables incl "
        f"documents_raw, doc_clusters, doc_chunks, entity_drugs, link_drug_*, "
        f"entity_proteins, clinical_trials, binding_affinities, ensembl_genes). Keep "
        f"queries few and targeted — look specifically for asymmetries (undrugged but "
        f"structurally-studied targets, drugs active across unrelated conditions, "
        f"research-rich untrialed drugs). Then write a NOVEL-INSIGHT strategy memo: novel "
        f"cross-connections, druggable/translational whitespace, testable hypotheses, a "
        f"contrarian angle, open questions. Do NOT summarize papers or restate known "
        f"facts. Output ONLY the memo. Never invent numbers.")
    cmd = ["claude", "-p", task, "--model", cfg["pro"], "--max-turns", str(max_turns),
           "--add-dir", str(sandbox),
           "--allowedTools", "Read", f"Bash(python {qpy.name}:*)", f"Bash(echo:*)"]
    try:
        r = subprocess.run(cmd, cwd=sandbox, env=env, capture_output=True,
                           text=True, timeout=420)
        return r.stdout.strip() or f"(agent produced no output; stderr: {r.stderr[:200]})"
    except subprocess.TimeoutExpired:
        return "(agent timed out)"


def _compose(topic: str | None, narrative: str, sheet: str, cites: list[tuple]) -> str:
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = f"Aqueduct report — {topic}" if topic else "Aqueduct landscape report"
    md = [f"# {title}", f"_Generated {when}_\n", narrative or "_(no narrative)_"]
    if cites:
        md.append("\n## Sources\n" + "\n".join(f"- `{p}` — {t or ''}" for p, t in cites))
    md.append("\n---\n## Data appendix (computed metrics)\n\n" + sheet)
    return "\n".join(md)


def _save(topic: str | None, md: str) -> Path:
    out_dir = config.DATA_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = (topic or "landscape").lower().replace(" ", "-")[:40]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{stamp}_{slug}.md"
    path.write_text(md)
    return path


def generate(topic: str | None = None, *, agent: bool = False, con=None,
             model: str = "pro", email: bool = False, to: str | None = None) -> dict:
    """Produce a grounded report. Returns {'path', 'markdown', 'mode', 'emailed'}."""
    owns = con is None
    con = con or connect()
    try:
        sheet = analysis.facts_sheet(con)
        snippets, cites = _snippets(topic, con=con) if topic else ("", [])
        if not llm.available():
            narrative, mode = "_(LLM unavailable — facts-only report)_", "facts-only"
        elif agent:
            narrative, mode = _agentic(sheet, snippets, topic), "agent"
        else:
            narrative = llm.complete(_prompt(sheet, snippets, topic), system=SYSTEM,
                                     model=model, max_tokens=4000)
            mode = "single-call"
        md = _compose(topic, narrative, sheet, cites)
        path = _save(topic, md)
        print(f"[report]  {mode} -> {path}")
        emailed = None
        if email:
            subject = f"Aqueduct report — {topic}" if topic else "Aqueduct landscape report"
            try:
                emailed = mailer.send(subject, md, to=to)
                print(f"[report]  emailed (id {emailed})")
            except mailer.MailUnavailable as e:
                print(f"[report]  email skipped: {e}")
        return {"path": str(path), "markdown": md, "mode": mode, "emailed": emailed}
    finally:
        if owns:
            con.close()
