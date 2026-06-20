#!/usr/bin/env bash
# Aqueduct daily report — generate a novelty strategy memo and email it.
# One DeepSeek call per run (token-prudent). Scheduled separately from the harvest.

PROJECT="/root/projects/aqueduct"
cd "$PROJECT" || exit 1

# Secrets: RESEND_API_KEY, AQUEDUCT_EMAIL_TO (+ optional REPORT_TOPIC).
[ -f "$PROJECT/.secrets.env" ] && . "$PROJECT/.secrets.env"
export OPENALEX_MAILTO="${OPENALEX_MAILTO:-work@supercriticalbooks.com}"

LOG="$PROJECT/data/report.log"
mkdir -p "$PROJECT/data"

echo "=== $(date -Is) report start ===" >>"$LOG"
if [ -n "${REPORT_TOPIC:-}" ]; then
  "$PROJECT/.venv/bin/python" -m aqueduct report --topic "$REPORT_TOPIC" --email >>"$LOG" 2>&1
else
  "$PROJECT/.venv/bin/python" -m aqueduct report --email >>"$LOG" 2>&1
fi
rc=$?
echo "=== $(date -Is) report done (exit $rc) ===" >>"$LOG"
tail -n 2000 "$LOG" > "$LOG.tmp" 2>/dev/null && mv "$LOG.tmp" "$LOG"
exit $rc
