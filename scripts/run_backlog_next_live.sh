#!/usr/bin/env bash
# Start backlog if needed, then stream the log in this terminal.
# Ctrl+C stops only tail; the backlog keeps running (use durable PID to stop).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

LOG="${REPO}/data/phase1/overnight/backlog_next_latest.log"
PROGRESS="${REPO}/data/phase1/overnight/backlog_next_latest_progress.txt"
PIDFILE="${REPO}/data/phase1/overnight/backlog_next.pid"

bash "${REPO}/scripts/run_backlog_next_durable.sh"

echo ""
echo "==> Live log: $LOG"
echo "==> Steps:    $PROGRESS"
echo "==> PID:      $(cat "$PIDFILE" 2>/dev/null || echo '?')"
echo "    Ctrl+C = stop watching only (backlog keeps running)"
echo "    Stop job: kill \$(cat $PIDFILE)"
echo ""

# Show recent steps, then follow MEEP output.
if [[ -f "$PROGRESS" ]]; then
  echo "--- progress (last steps) ---"
  tail -8 "$PROGRESS"
  echo "-----------------------------"
  echo ""
fi

touch "$LOG"
tail -f "$LOG"
