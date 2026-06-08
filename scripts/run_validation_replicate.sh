#!/usr/bin/env bash
# One MEEP validation replicate (run_06) at B=100 with promoted surrogate_improved.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
source .venv/bin/activate

CONFIG="${CONFIG:-configs/wedge_a_validate_replicate.yaml}"
REPLICATE_ID="${REPLICATE_ID:-6}"
BASE_SEED="${BASE_SEED:-2026}"
SEED=$((BASE_SEED + REPLICATE_ID * 1000))

SUR="data/phase1/wedge_a/surrogate_improved"
if [[ ! -f "$SUR/surrogate.joblib" ]]; then
  echo "ERROR: train improved surrogate first:"
  echo "  python scripts/train_wedge_a_surrogate.py --config configs/wedge_a_improved.yaml"
  exit 1
fi

REPORT="data/phase1/wedge_a/sim_budget/replicates/run_$(printf '%02d' "$REPLICATE_ID")/run_report.json"
if [[ -f "$REPORT" ]]; then
  echo "==> Skip validation replicate run_${REPLICATE_ID}: $REPORT exists"
  exit 0
fi

echo "==> Validation replicate run_${REPLICATE_ID} seed=$SEED config=$CONFIG"
bash scripts/run_meep.sh scripts/run_sim_budget_study.py \
  --config "$CONFIG" \
  --replicate-id "$REPLICATE_ID" \
  --seed "$SEED" \
  --only-missing \
  --ensure-candidates

echo ""
echo "==> Compare run_${REPLICATE_ID}/run_report.json to run_01..05 at B=100"
echo "    data/phase1/wedge_a/sim_budget/replicates/run_${REPLICATE_ID}/"
