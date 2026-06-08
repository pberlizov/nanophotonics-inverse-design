#!/usr/bin/env bash
# Full Wedge A sim-budget study: MEEP budgets 30, 50, 100 (all policies).
# Requires conda env mp — see docs/MEEP_SETUP.md.
#
#   bash scripts/run_full_sim_budget.sh
#   bash scripts/run_full_sim_budget.sh --skip-train
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
source .venv/bin/activate

SKIP_TRAIN=0
[[ "${1:-}" == "--skip-train" ]] && SKIP_TRAIN=1

if [[ "$SKIP_TRAIN" != 1 ]]; then
  echo "==> Train wedge-A ranker"
  python scripts/train_wedge_a_surrogate.py --config configs/wedge_a.yaml
fi

echo "==> Sim-budget MEEP (budgets 30, 50, 100) — this may take several hours"
SUR="${SIM_BUDGET_SURROGATE:-data/phase1/wedge_a/surrogate_mask_perturb}"
if [[ ! -f "$SUR/surrogate.joblib" ]]; then
  SUR="data/phase1/wedge_a/surrogate"
fi
python scripts/generate_ranked_candidates.py \
  --config configs/wedge_a.yaml \
  --surrogate "$SUR" \
  --n-proposals 2500 \
  --output data/phase1/wedge_a/sim_budget/candidates_pool.csv

bash scripts/run_meep.sh scripts/run_sim_budget_study.py \
  --config configs/wedge_a.yaml \
  --budgets 30 50 100 \
  --only-missing \
  --replace-run

echo "==> Regenerate pilot collateral"
bash scripts/run_pilot.sh --config configs/pilot/benchmark_50_50.yaml --report-only

echo "Done. Check data/phase1/wedge_a/wedge_a_metrics.json for budgets 50 and 100."
