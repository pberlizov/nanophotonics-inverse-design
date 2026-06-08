#!/usr/bin/env bash
# Sprint v2 launcher — morph / broadband / IL tracks with CPU-aware staggering.
#
#   bash scripts/sprint_launch.sh              # print plan + status
#   bash scripts/sprint_launch.sh verify       # Phase2 top-10 fine-grid verify (~1 h)
#   bash scripts/sprint_launch.sh morph        # morph sprint v2 (~4–6 h)
#   bash scripts/sprint_launch.sh broadband    # refine + explore (~4–8 h)
#   bash scripts/sprint_launch.sh il           # IL sprint v2 (~3–5 h)
#   bash scripts/sprint_launch.sh stagger      # verify now, morph +2h, il +4h (background)
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
LOG="$REPO/data/phase1/release/sprint_logs"
mkdir -p "$LOG" data/phase1/sprint_broadband_v2

MEEP="bash scripts/run_meep.sh"
PY=".venv/bin/python"

cpu_note() {
  if pgrep -f "run_release_replication" >/dev/null 2>&1; then
    echo "NOTE: replication (N_REPLICATES=20) is running — sprint jobs will contend for CPU."
  fi
  echo "Logical CPUs: $(sysctl -n hw.ncpu 2>/dev/null || echo '?')"
}

cmd_verify() {
  echo "$MEEP scripts/broadband_rescore_candidates.py \\"
  echo "  --candidates data/phase1/phase2_broadband_hunt/explore/top_candidates.csv \\"
  echo "  --wl-start 1.53 --wl-stop 1.57 --wl-step 0.005 \\"
  echo "  --max-worst-split-error 0.05 --limit 10 \\"
  echo "  --output data/phase1/sprint_broadband_v2/phase2_top10_verify.json"
}

cmd_morph() {
  echo "$MEEP scripts/phase2_morph_hunt.py --config configs/sprint_morph_v2.yaml"
}

cmd_broadband_refine() {
  echo "$MEEP scripts/broadband_refine_from_centers.py \\"
  echo "  --config configs/sprint_broadband_v2.yaml \\"
  echo "  --output-dir data/phase1/sprint_broadband_v2/refine \\"
  echo "  --trials-per-center 50"
}

cmd_broadband_explore() {
  echo "$MEEP scripts/latent_meep_search.py \\"
  echo "  --config configs/sprint_broadband_v2.yaml \\"
  echo "  --objective broadband --latent-mode perlin --n-trials 100 \\"
  echo "  --output-dir data/phase1/sprint_broadband_v2/explore"
}

cmd_il() {
  echo "$MEEP scripts/phase2_il_hunt.py --config configs/sprint_il_v2.yaml"
}

run_verify() {
  mkdir -p data/phase1/sprint_broadband_v2
  nohup bash -lc "cd $REPO && $MEEP scripts/broadband_rescore_candidates.py \
    --candidates data/phase1/phase2_broadband_hunt/explore/top_candidates.csv \
    --wl-start 1.53 --wl-stop 1.57 --wl-step 0.005 \
    --max-worst-split-error 0.05 --limit 10 \
    --output data/phase1/sprint_broadband_v2/phase2_top10_verify.json" \
    > "$LOG/sprint_bb_verify.log" 2>&1 &
  echo "broadband verify PID $!  log: $LOG/sprint_bb_verify.log"
}

run_morph() {
  nohup bash -lc "cd $REPO && $MEEP scripts/phase2_morph_hunt.py --config configs/sprint_morph_v2.yaml" \
    > "$LOG/sprint_morph_v2.log" 2>&1 &
  echo "morph sprint PID $!  log: $LOG/sprint_morph_v2.log"
}

run_broadband() {
  nohup bash -lc "cd $REPO && \
    $MEEP scripts/broadband_refine_from_centers.py \
      --config configs/sprint_broadband_v2.yaml \
      --output-dir data/phase1/sprint_broadband_v2/refine \
      --trials-per-center 50 && \
    $MEEP scripts/latent_meep_search.py \
      --config configs/sprint_broadband_v2.yaml \
      --objective broadband --latent-mode perlin --n-trials 100 \
      --output-dir data/phase1/sprint_broadband_v2/explore" \
    > "$LOG/sprint_broadband_v2.log" 2>&1 &
  echo "broadband sprint PID $!  log: $LOG/sprint_broadband_v2.log"
}

run_il() {
  nohup bash -lc "cd $REPO && $MEEP scripts/phase2_il_hunt.py --config configs/sprint_il_v2.yaml" \
    > "$LOG/sprint_il_v2.log" 2>&1 &
  echo "IL sprint PID $!  log: $LOG/sprint_il_v2.log"
}

run_stagger() {
  run_verify
  (
    sleep 7200
    run_morph
  ) &
  echo "morph scheduled +2h (PID $!)"
  (
    sleep 14400
    run_il
  ) &
  echo "IL scheduled +4h (PID $!)"
  echo "replication left running; broadband refine/explore: launch manually after verify"
  echo "  bash scripts/sprint_launch.sh broadband"
}

ACTION="${1:-}"

if [[ -z "$ACTION" ]]; then
  cpu_note
  cat <<EOF

Sprint v2 — three parallel tracks (see docs/SPRINT_MORPH_BB_IL.md)

Track A — Morph robustness (configs/sprint_morph_v2.yaml)
  Gate: morph_pass=True (worst_split≤0.05, max_delta≤0.05 @ 10/15/20nm)
  $(cmd_morph)

Track B — Broadband C-band (configs/sprint_broadband_v2.yaml)
  Step 1 — verify Phase2 explore top-10 @ 5nm:
  $(cmd_verify)
  Step 2 — residual refine from 00182/00101:
  $(cmd_broadband_refine)
  Step 3 — Perlin explore (100 trials):
  $(cmd_broadband_explore)
  Gate: worst |R_up−0.5| ≤ 0.05 over 1.53–1.57 µm (5nm grid)

Track C — IL/T (configs/sprint_il_v2.yaml)
  Prereq: matgrid calibration done (data/phase1/meep_research/matgrid_calibration_combined.json)
  $(cmd_il)
  Gate: split_err≤0.05 AND IL≤12 dB (same design)

Launch:
  bash scripts/sprint_launch.sh verify
  bash scripts/sprint_launch.sh stagger    # recommended with replication running
  bash scripts/sprint_launch.sh morph|broadband|il

Logs: $LOG/
EOF
  exit 0
fi

cpu_note
case "$ACTION" in
  verify) run_verify ;;
  morph) run_morph ;;
  broadband) run_broadband ;;
  il) run_il ;;
  stagger) run_stagger ;;
  *)
    echo "Unknown action: $ACTION (verify|morph|broadband|il|stagger)"
    exit 1
    ;;
esac

echo "tail -f $LOG/sprint_*.log"
