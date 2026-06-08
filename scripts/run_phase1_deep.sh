#!/usr/bin/env bash
# Phase 1 deep dev: multi-objective MEEP search, surrogate-ranked AL, layout export.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate

echo "==> 0/6 Champion export (PNG + GDS if gdstk installed)"
python scripts/export_best_design.py \
  --sample-id local_00022 \
  --mask data/phase1/meep_search_local/candidates/masks/local_00022_mask.npy \
  --output-dir data/phase1/exports
if python -c "import gdstk" 2>/dev/null; then
  python scripts/export_layout_gds.py \
    --sample-id local_00022 \
    --mask data/phase1/meep_search_local/candidates/masks/local_00022_mask.npy \
    --output-dir data/phase1/exports
else
  echo "    (skip GDS — pip install gdstk)"
fi

echo "==> 1/6 Optional: calibrate phase0_v2 template spike"
bash scripts/run_meep.sh scripts/calibrate_meep.py \
  --config configs/phase0.yaml --recipe-version phase0_v2 --verbose || true

echo "==> 2/6 Global MEEP search (multi-objective, 150 trials)"
bash scripts/run_meep.sh scripts/meep_search.py \
  --config configs/phase1.yaml \
  --output-dir data/phase1/meep_search_deep \
  --n-trials 150 \
  --objective multi \
  --sigma-min 0.008 \
  --sigma-max 0.035

echo "==> 3/6 Local sigma refine around champion"
bash scripts/run_meep.sh scripts/meep_search_local.py \
  --config configs/phase1.yaml \
  --center-sigma 0.0139193747656368 \
  --n-trials 50 \
  --output-dir data/phase1/meep_search_local_deep

echo "==> 4/6 Surrogate-ranked AL round 1 (presearch -> MEEP verify -> retrain)"
python scripts/surrogate_ranked_al_round.py --round 1

echo "==> 5/6 Re-ranking eval on expanded corpus"
python scripts/evaluate_surrogate_ranking.py \
  --surrogate data/phase1/al_round_01/surrogate \
  --sim-results data/phase0/sim_results_phase0_v1_all.csv \
  --sources all \
  --output data/phase1/al_round_01/ranking_eval.json

echo "Done. See docs/PHASE1_DEEP_DEV.md and data/phase1/"
