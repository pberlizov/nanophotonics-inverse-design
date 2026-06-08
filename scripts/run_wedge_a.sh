#!/usr/bin/env bash
# Wedge A — Tier 1 pilot: train ranker → ranking gate → sim-budget study (pilot budgets).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate

PILOT=1
[[ "${1:-}" == "--full" ]] && { PILOT=0; shift; }

echo "==> 1/3 Train perturb latent_mlp + ranking gate"
python scripts/train_wedge_a_surrogate.py --config configs/wedge_a.yaml

echo "==> 2/3 Sim-budget study (MEEP)"
if [[ "$PILOT" == 1 ]]; then
  bash scripts/run_meep.sh scripts/run_sim_budget_study.py \
    --config configs/wedge_a.yaml --pilot --ensure-candidates
else
  bash scripts/run_meep.sh scripts/run_sim_budget_study.py \
    --config configs/wedge_a.yaml --ensure-candidates
fi

echo "==> 3/3 Optional: one acquisition round (skip with SKIP_ROUND=1)"
if [[ "${SKIP_ROUND:-}" != "1" ]]; then
  python scripts/run_wedge_a_round.py --round 1
fi

echo "Done. See data/phase1/wedge_a/wedge_a_metrics.json and docs/WEDGE_A.md"
