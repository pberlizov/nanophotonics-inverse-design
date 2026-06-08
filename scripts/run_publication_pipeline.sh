#!/usr/bin/env bash
# Single-publication pipeline: v1 finalize → IL hunt → report → PDF.
#
#   bash scripts/run_publication_pipeline.sh                  # print plan
#   bash scripts/run_publication_pipeline.sh --loss-aware-only
#   bash scripts/run_publication_pipeline.sh --finalize-only
#   bash scripts/run_publication_pipeline.sh --launch-loss-aware  # background MEEP
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
LOG="$REPO/data/phase1/release/followup_logs"
mkdir -p "$LOG"
PY=".venv/bin/python"

if [[ "${1:-}" == "--launch-loss-aware" ]]; then
  echo "Launching loss-aware search (background)..."
  nohup bash -lc "cd $REPO && source .venv/bin/activate && bash scripts/run_meep.sh scripts/loss_aware_search.py --trials-per-center 40" \
    > "$LOG/loss_aware_search.log" 2>&1 &
  echo "PID $! — tail -f $LOG/loss_aware_search.log"
  exit 0
fi

if [[ "${1:-}" == "--loss-aware-only" ]]; then
  bash scripts/run_meep.sh scripts/loss_aware_search.py --trials-per-center 40
  $PY scripts/loss_aware_report.py
  $PY scripts/export_loss_aware_figure.py 2>/dev/null || true
  bash scripts/finalize_preprint_v1.sh
  $PY scripts/check_publication_readiness.py
  exit 0
fi

if [[ "${1:-}" == "--finalize-only" ]]; then
  bash scripts/finalize_preprint_v1.sh
  $PY scripts/check_publication_readiness.py
  exit 0
fi

cat <<EOF
Publication pipeline (single arXiv upload)
See docs/preprint/PUBLICATION_PLAN.md

Phase A — v1 engineering (terminals in flight):
  - n=20 replication, broadband hunt, morph-robust
  - When done: bash scripts/finalize_preprint_v1.sh

Phase B — IL objective (required before upload):
  bash scripts/run_publication_pipeline.sh --launch-loss-aware
  # or foreground:
  bash scripts/run_publication_pipeline.sh --loss-aware-only

Phase C — upload:
  python scripts/check_publication_readiness.py
  # update manuscript §loss-aware, then arXiv

EOF
