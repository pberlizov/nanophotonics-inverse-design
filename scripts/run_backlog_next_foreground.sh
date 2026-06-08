#!/usr/bin/env bash
# Run backlog in the foreground with live stdout + persistent log (for IDE terminal).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

LOG_DIR="${REPO}/data/phase1/overnight"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOG_DIR}/backlog_foreground_${STAMP}.log"
ln -sfn "$(basename "$LOG")" "${LOG_DIR}/backlog_foreground_latest.log"

echo "==> Backlog (foreground) — log: $LOG"
echo "==> Progress: ${LOG_DIR}/backlog_next_latest_progress.txt"
echo "==> Press Ctrl+C to stop (matgrid --resume is safe to restart)"
echo ""

# Unbuffered Python/tqdm; tee for live view. pipefail off so closing the
# terminal does not SIGPIPE-kill MEEP mid-run (log still written via exec >> in child).
export PYTHONUNBUFFERED=1
set +o pipefail
bash "${REPO}/scripts/run_backlog_next.sh" 2>&1 | tee -a "$LOG"
