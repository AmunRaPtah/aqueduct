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

# Embedding backend for the unattended build. Default to the keyless LSA path: on this
# memory-tight box it peaks ~400 MB and rebuilds in ~90s, whereas sentence-transformers
# (~1.5 GB, ~30 min for a full re-embed) risks OOM under concurrent cron load and can
# outrun the watchdog. Set AQUEDUCT_EMBED_BACKEND=st for higher-quality manual runs.
export AQUEDUCT_EMBED_BACKEND="${AQUEDUCT_EMBED_BACKEND:-lsa}"

TOPICS="${TOPICS:-$PROJECT/topics.json}"
# Per-query page size PER RUN. Ingestors now persist a pagination cursor in
# harvest_state.json, so each hourly run pages *deeper* into each source instead of
# re-reading the newest page. A smaller page keeps every run cheap (less full-text
# efetch / PDF work → no more 45m timeouts) while the corpus still grows every hour.
LIMIT="${LIMIT:-15}"
# Hard wall-clock cap so a hung source (network/backoff) can never wedge the lock
# past the next hourly run. Exit 124 = timed out. Kept under the 1h cron interval.
TIMEOUT="${TIMEOUT:-45m}"
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
# Hard memory ceiling via a transient cgroup scope: if the harvest balloons (e.g. an
# accidental `st` embed backend, or a runaway source), it is killed inside its OWN
# cgroup instead of triggering a global OOM that freezes the box.
# 2026-06-23: the index/embed build on the grown corpus peaks just over the old 1.5G
# RAM + 512M swap cap and was being SIGKILLed (exit 137) every run before completing.
# Bumped RAM ceiling to 2G and swap allowance to 2G so it spills to (compressed zram)
# swap and FINISHES, while still capped well below box limits. MemoryMax keeps real
# RAM bounded; MemorySwapMax stops it eating the whole swap pool.
systemd-run --scope --quiet --collect -p MemoryMax=2G -p MemorySwapMax=2G \
  timeout --signal=TERM --kill-after=60 "$TIMEOUT" \
  "$PROJECT/.venv/bin/python" -m aqueduct harvest --topics "$TOPICS" --limit "$LIMIT" >>"$LOG" 2>&1
rc=$?
[ "$rc" = 124 ] && echo "$(date -Is) harvest TIMED OUT after $TIMEOUT (watchdog killed it)" >>"$LOG"
echo "=== $(date -Is) harvest done (exit $rc) ===" >>"$LOG"

# Keep the log bounded (last 5000 lines).
tail -n 5000 "$LOG" > "$LOG.tmp" 2>/dev/null && mv "$LOG.tmp" "$LOG"
exit $rc
