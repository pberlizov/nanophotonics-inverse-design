#!/usr/bin/env bash
# Phase 2 campaigns: IL (A,B,C), C-band (A,B,C,D), Morph (B,C,D).
#
#   bash scripts/run_phase2_campaigns.sh           # print plan
#   bash scripts/run_phase2_campaigns.sh launch    # background MEEP jobs
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
LOG="$REPO/data/phase1/release/followup_logs"
mkdir -p "$LOG"

if [[ "${1:-}" != "launch" ]]; then
  cat <<EOF
Phase 2 MEEP campaigns (use 3 terminals or launch all):

  # IL — A) weight_il=0.75  B) stage-2 IL refine  C) Perlin/sigma seeds
  bash scripts/run_meep.sh scripts/phase2_il_hunt.py

  # C-band — A) Perlin scratch  B) flatness=0.5  C) λ step 5nm  D) 200 trials
  bash scripts/run_meep.sh scripts/latent_meep_search.py \\
    --config configs/phase2_broadband_hunt.yaml \\
    --objective broadband --latent-mode perlin --n-trials 200 \\
    --output-dir data/phase1/phase2_broadband_hunt/explore

  # Morph — B) Perlin  C) dilate-survivor centers  D) asymmetric loss
  bash scripts/run_meep.sh scripts/phase2_morph_hunt.py

  bash scripts/run_phase2_campaigns.sh launch
EOF
  exit 0
fi

echo "Launching Phase 2 campaigns (logs in $LOG)..."
nohup bash -lc "cd $REPO && source .venv/bin/activate && bash scripts/run_meep.sh scripts/phase2_il_hunt.py" \
  > "$LOG/phase2_il_hunt.log" 2>&1 &
echo "phase2 IL PID $!"

nohup bash -lc "cd $REPO && source .venv/bin/activate && bash scripts/run_meep.sh scripts/latent_meep_search.py --config configs/phase2_broadband_hunt.yaml --objective broadband --latent-mode perlin --n-trials 200 --output-dir data/phase1/phase2_broadband_hunt/explore" \
  > "$LOG/phase2_broadband_hunt.log" 2>&1 &
echo "phase2 broadband PID $!"

nohup bash -lc "cd $REPO && source .venv/bin/activate && bash scripts/run_meep.sh scripts/phase2_morph_hunt.py" \
  > "$LOG/phase2_morph_hunt.log" 2>&1 &
echo "phase2 morph PID $!"

echo "tail -f $LOG/phase2_*.log"
