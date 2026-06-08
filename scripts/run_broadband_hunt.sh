#!/usr/bin/env bash
# Full broadband contribution hunt (~2–6 h MEEP depending on trials).
#
# Produces:
#   data/phase1/broadband_hunt/          — search artifacts
#   data/phase1/release/broadband_hunt.md — winners + paper hooks
#   data/phase1/release/broadband_contribution.png — before/after figure
#
# Quick pilot (skip explore, fewer trials):
#   TRIALS_PER_CENTER=10 SKIP_EXPLORE=1 bash scripts/run_broadband_hunt.sh
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

TRIALS_PER_CENTER="${TRIALS_PER_CENTER:-25}"
EXPLORE_TRIALS="${EXPLORE_TRIALS:-40}"
SKIP_EXPLORE="${SKIP_EXPLORE:-0}"

echo "==> [1/6] Build in-spec candidate pool (CPU)"
.venv/bin/python scripts/broadband_hunt.py --stage pool

echo "==> [2/6] Coarse rescore on pool (MEEP; --resume if partial JSON exists)"
bash scripts/run_meep.sh scripts/broadband_rescore_candidates.py \
  --candidates data/phase1/broadband_hunt/candidate_pool.csv \
  --config configs/broadband_hunt.yaml \
  --wl-step 0.01 \
  --max-worst-split-error 0.05 \
  --output data/phase1/broadband_hunt/pool_broadband_rescore.json \
  --resume

echo "==> [3/6] Refine from champion centers (MEEP, ${TRIALS_PER_CENTER} trials × 4 centers)"
bash scripts/run_meep.sh scripts/broadband_refine_from_centers.py \
  --trials-per-center "$TRIALS_PER_CENTER"

if [[ "$SKIP_EXPLORE" != "1" ]]; then
  echo "==> [4/6] Latent explore (MEEP, ${EXPLORE_TRIALS} trials)"
  bash scripts/run_meep.sh scripts/latent_meep_search.py \
    --config configs/broadband_hunt.yaml \
    --output-dir data/phase1/broadband_hunt/explore \
    --objective broadband \
    --latent-mode residual \
    --n-trials "$EXPLORE_TRIALS"
else
  echo "==> [4/6] Latent explore skipped"
fi

echo "==> [5/6] Fine-grid verify top candidates (MEEP)"
bash scripts/run_meep.sh scripts/broadband_hunt.py --stage verify

echo "==> [6/6] Report + contribution figure (CPU)"
.venv/bin/python scripts/broadband_hunt.py --stage report
.venv/bin/python scripts/export_broadband_contribution_figure.py

echo ""
echo "Done. Read data/phase1/release/broadband_hunt.md"
echo "If n_winners=0, try TRIALS_PER_CENTER=50 or add flatness_weight in configs/broadband_hunt.yaml"
