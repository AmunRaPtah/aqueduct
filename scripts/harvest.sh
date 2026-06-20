#!/usr/bin/env bash
# Aqueduct scheduled harvest — refreshes the local database from a topics file.
# Safe to run from cron: sets identifiers, prevents overlapping runs, logs to file.

PROJECT="/root/projects/aqueduct"
cd "$PROJECT" || exit 1

# Local secrets (RESEND_API_KEY, AQUEDUCT_EMAIL_TO, PATENTSVIEW_API_KEY) — untracked.
[ -f "$PROJECT/.secrets.env" ] && . "$PROJECT/.secrets.env"

# Polite-pool identifiers (APIs ask for a contact).
export NCBI_EMAIL="${NCBI_EMAIL:-work@supercriticalbooks.com}"
export OPENALEX_MAILTO="${OPENALEX_MAILTO:-work@supercriticalbooks.com}"

TOPICS="${TOPICS:-$PROJECT/topics.json}"
LIMIT="${LIMIT:-25}"
LOG="$PROJECT/data/harvest.log"
mkdir -p "$PROJECT/data"

# One run at a time — skip if a previous harvest is still going.
exec 9>"$PROJECT/data/.harvest.lock"
if ! flock -n 9; then
  echo "$(date -Is) harvest already running — skipping" >>"$LOG"
  exit 0
fi

if [ ! -f "$TOPICS" ]; then
  echo "$(date -Is) no topics file at $TOPICS (cp topics.example.json topics.json)" >>"$LOG"
  exit 0
fi

echo "=== $(date -Is) harvest start (topics=$TOPICS limit=$LIMIT) ===" >>"$LOG"
"$PROJECT/.venv/bin/python" -m aqueduct harvest --topics "$TOPICS" --limit "$LIMIT" >>"$LOG" 2>&1
rc=$?
echo "=== $(date -Is) harvest done (exit $rc) ===" >>"$LOG"

# Keep the log bounded (last 5000 lines).
tail -n 5000 "$LOG" > "$LOG.tmp" 2>/dev/null && mv "$LOG.tmp" "$LOG"
exit $rc
