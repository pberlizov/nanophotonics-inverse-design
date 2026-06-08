#!/usr/bin/env bash
# Resume deep pipeline after step 2 (global search already done).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate

echo "==> 3/6 Local sigma refine around champion"
bash scripts/run_meep.sh scripts/meep_search_local.py \
  --config configs/phase1.yaml \
  --center-sigma 0.0139193747656368 \
  --n-trials 50 \
  --output-dir data/phase1/meep_search_local_deep

echo "==> 4/6 Surrogate-ranked AL round 1"
python scripts/surrogate_ranked_al_round.py --round 1

echo "==> 5/6 Re-ranking eval on expanded corpus"
python scripts/evaluate_surrogate_ranking.py \
  --surrogate data/phase1/al_round_01/surrogate \
  --sim-results data/phase0/sim_results_phase0_v1_all.csv \
  --sources all \
  --output data/phase1/al_round_01/ranking_eval.json

echo "Done (resume). See data/phase1/"
