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
# past the next run. Exit 124 = timed out. Cron here is daily (not hourly, despite
# the old comment above) and flock already blocks overlap, so there's no reason to
# cap this tighter than the corpus actually needs. 45m was killing every run before
# a full pass finished (162 queries across 10 sources, incl. 240 europepmc hits/run,
# far more than leviathan/prometheus's ~2 sources each). Bumped to 3h, still well
# under the 24h cadence.
TIMEOUT="${TIMEOUT:-3h}"
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
# 2026-07-04 (post-wipe rebuild): this box has NO swap configured (Swap: 0B), so the
# 2G+2G RAM/swap design could not spill and was SIGKILLed (exit 137) at ~2.1G every run.
# Give it the full 2G+2G working-set budget as real RAM (4G) instead — box has ~7.8G,
# so 4G is still safely below box limits. If swap is later restored, MemorySwapMax reapplies.
systemd-run --scope --quiet --collect -p MemoryMax=4G -p MemorySwapMax=2G \
  timeout --signal=TERM --kill-after=60 "$TIMEOUT" \
  "$PROJECT/.venv/bin/python" -m aqueduct harvest --topics "$TOPICS" --limit "$LIMIT" >>"$LOG" 2>&1
rc=$?
[ "$rc" = 124 ] && echo "$(date -Is) harvest TIMED OUT after $TIMEOUT (watchdog killed it)" >>"$LOG"
echo "=== $(date -Is) harvest done (exit $rc) ===" >>"$LOG"

# Always resync the semantic index, even when the ingest above timed out.
# `harvest` only reaches its own build_index call after the slow document-fetch
# phase finishes. If the 3h watchdog kills it mid-ingest (as happened on every run
# 2026-10-18..22), the index is silently left stale while doc_chunks keeps growing,
# so every `rag`/`semantic` query then scores against months-old chunk ids and
# returns nothing. This step runs unconditionally so the index always reflects
# whatever chunks actually landed this run. `build_index` auto-skips (hashes
# corpus) if nothing changed, so a clean run pays only a hash check, not a re-embed.
echo "$(date -Is) post-harvest index resync (rc=$rc)" >>"$LOG"
nice -n 19 ionice -c3 systemd-run --scope --quiet --collect -p MemoryMax=2G -p MemorySwapMax=2G \
  timeout --signal=TERM --kill-after=30 10m \
  "$PROJECT/.venv/bin/python" -m aqueduct corpus index --backend "$AQUEDUCT_EMBED_BACKEND" >>"$LOG" 2>&1
irc=$?
[ "$irc" = 124 ] && echo "$(date -Is) index resync TIMED OUT after 10m" >>"$LOG"
echo "$(date -Is) index resync done (exit $irc)" >>"$LOG"

# Keep the log bounded (last 5000 lines).
tail -n 5000 "$LOG" > "$LOG.tmp" 2>/dev/null && mv "$LOG.tmp" "$LOG"
exit $rc
