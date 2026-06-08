#!/usr/bin/env bash
# Detached backlog runner — survives closing Cursor chat/terminals (use this for long MEEP).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

LOG_DIR="${REPO}/data/phase1/overnight"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOG_DIR}/backlog_durable_${STAMP}.log"
PIDFILE="${LOG_DIR}/backlog_next.pid"

if [[ -f "$PIDFILE" ]]; then
  OLD_PID="$(cat "$PIDFILE")"
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Already running PID=${OLD_PID}."
    echo "  Watch live: bash scripts/run_backlog_next_live.sh"
    echo "  Log:        $LOG_DIR/backlog_next_latest.log"
    exit 0
  fi
  echo "Stale PID file (${OLD_PID} not running) — starting fresh."
  rm -f "$PIDFILE"
fi

echo "Starting durable backlog → $LOG"
echo "  progress: $LOG_DIR/backlog_next_latest_progress.txt"
echo "  stop: kill \$(cat $PIDFILE)"

# setsid + nohup: new session, no controlling tty (survives terminal/Cursor tab close).
if command -v setsid >/dev/null 2>&1; then
  WRAP=(setsid nohup bash -c)
else
  WRAP=(nohup bash -c)
fi
"${WRAP[@]}" "
  cd \"$REPO\"
  export PYTHONUNBUFFERED=1
  exec bash scripts/run_backlog_next.sh
" </dev/null >>"$LOG" 2>&1 &

echo $! >"$PIDFILE"
ln -sfn "$(basename "$LOG")" "${LOG_DIR}/backlog_durable_latest.log"
echo "PID=$(cat "$PIDFILE")"
