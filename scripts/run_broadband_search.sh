#!/usr/bin/env bash
# Broadband-aware local σ search + verification.
#
# Step 1 (MEEP, ~hours): Optuna with coarse 5-point C-band objective
# Step 2 (MEEP, ~minutes): Fine-grid rescore on top candidates
# Step 3 (MEEP, optional): Re-sweep promoted champions at 0.005 µm step
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

N_TRIALS="${N_TRIALS:-30}"
OUT_DIR="${OUT_DIR:-data/phase1/meep_search_broadband}"

echo "==> [1/3] Broadband local search (n_trials=$N_TRIALS)"
bash scripts/run_meep.sh scripts/meep_search_local.py \
  --config configs/broadband_search_local.yaml \
  --output-dir "$OUT_DIR" \
  --objective broadband \
  --n-trials "$N_TRIALS"

echo "==> [2/3] Fine-grid rescore on top candidates"
bash scripts/run_meep.sh scripts/broadband_rescore_candidates.py \
  --candidates "$OUT_DIR/top_candidates.csv" \
  --wl-step 0.005 \
  --output data/phase1/release/broadband_rescore.json

echo "==> [3/3] Champion verification (finer grid + gate)"
bash scripts/run_meep.sh scripts/champion_broadband_sweep.py \
  --wl-step 0.005 \
  --max-worst-split-error 0.05

echo "Done. Check data/phase1/release/broadband_rescore.md and champion_broadband.md"
