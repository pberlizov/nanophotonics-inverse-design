#!/usr/bin/env bash
# Extend sim-budget study with B=50 and B=100 only (skip completed runs).
# Requires conda env mp. Uses mask_mlp ranker if trained.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
source .venv/bin/activate

SUR="${SIM_BUDGET_SURROGATE:-data/phase1/wedge_a/surrogate_mask_perturb}"
if [[ ! -f "$SUR/surrogate.joblib" ]]; then
  echo "Using default surrogate (train mask_mlp first: python scripts/train_wedge_a_surrogate.py --config configs/wedge_a_mask.yaml)"
  SUR="data/phase1/wedge_a/surrogate"
fi

# Temporarily point wedge_a surrogate at best available ranker for candidate pool
export WEDGE_A_SURROGATE_OVERRIDE="$SUR"

echo "==> Ensure candidate pool sized for B=100"
python scripts/generate_ranked_candidates.py \
  --config configs/wedge_a.yaml \
  --surrogate "$SUR" \
  --n-proposals 2500 \
  --output data/phase1/wedge_a/sim_budget/candidates_pool.csv

echo "==> MEEP sim-budget B=50,100 (only missing)"
bash scripts/run_meep.sh scripts/run_sim_budget_study.py \
  --config configs/wedge_a.yaml \
  --budgets 50 100 \
  --only-missing \
  --replace-run

echo "==> Refresh pilot report"
bash scripts/run_pilot.sh --config configs/pilot/benchmark_50_50.yaml --report-only

echo "Done. Verify budgets 50,100 in data/phase1/wedge_a/wedge_a_metrics.json"
