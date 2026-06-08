#!/usr/bin/env bash
# Launch four follow-up workstreams (run each in its own terminal).
#
#   bash scripts/run_followup_priorities.sh          # print commands
#   bash scripts/run_followup_priorities.sh launch # background all four
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
LOG="${REPO}/data/phase1/release/followup_logs"
mkdir -p "$LOG"

cmd_replication() {
  echo "N_REPLICATES=20 CONFIG=configs/wedge_a_release_replication.yaml bash scripts/run_release_replication.sh"
}

cmd_broadband() {
  echo "bash scripts/run_broadband_hunt.sh"
}

cmd_flux() {
  echo "bash scripts/run_meep.sh scripts/flux_il_audit.py"
}

cmd_morph() {
  echo "bash scripts/run_meep.sh scripts/morph_robust_search.py --trials-per-center 30"
}

if [[ "${1:-}" != "launch" ]]; then
  cat <<EOF
Follow-up priorities — open four terminals and run:

Terminal A (replication, days):
  cd $REPO && source .venv/bin/activate
  $(cmd_replication) 2>&1 | tee $LOG/replication.log

Terminal B (broadband hunt, hours):
  cd $REPO && source .venv/bin/activate
  $(cmd_broadband) 2>&1 | tee $LOG/broadband_hunt.log

Terminal C (flux/IL audit, ~1 h):
  cd $REPO && source .venv/bin/activate
  $(cmd_flux) 2>&1 | tee $LOG/flux_il_audit.log

Terminal D (morph-robust search, ~2–4 h):
  cd $REPO && source .venv/bin/activate
  $(cmd_morph) 2>&1 | tee $LOG/morph_robust.log

After each finishes:
  bash scripts/finalize_preprint_v1.sh

Phase 1 (v2, parallel — not v1 blocking):
  bash scripts/run_meep.sh scripts/loss_aware_search.py

Or: bash scripts/run_followup_priorities.sh launch
EOF
  exit 0
fi

echo "Launching background jobs (logs in $LOG)..."
nohup bash -lc "cd $REPO && source .venv/bin/activate && $(cmd_replication)" > "$LOG/replication.log" 2>&1 &
echo "replication PID $!"
nohup bash -lc "cd $REPO && source .venv/bin/activate && $(cmd_broadband)" > "$LOG/broadband_hunt.log" 2>&1 &
echo "broadband PID $!"
nohup bash -lc "cd $REPO && source .venv/bin/activate && $(cmd_flux)" > "$LOG/flux_il_audit.log" 2>&1 &
echo "flux audit PID $!"
nohup bash -lc "cd $REPO && source .venv/bin/activate && $(cmd_morph)" > "$LOG/morph_robust.log" 2>&1 &
echo "morph-robust PID $!"
echo "tail -f $LOG/*.log"
