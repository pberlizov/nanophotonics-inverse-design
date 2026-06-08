#!/usr/bin/env bash
# Run N replicate sim-budget studies (frozen surrogate, varying seeds only).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
source .venv/bin/activate

N="${N_REPLICATES:-5}"
BASE_SEED="${BASE_SEED:-2026}"
CONFIG="${CONFIG:-configs/wedge_a_production.yaml}"

SUR=$(.venv/bin/python -c "
import yaml
from pathlib import Path
from nano_inv.pilot import load_pilot_config
cfg = load_pilot_config(Path('$CONFIG'))
print(cfg['surrogate']['output_dir'])
")
if [[ ! -f "$SUR/surrogate.joblib" ]]; then
  echo "ERROR: train production surrogate first: bash scripts/run_production_pipeline.sh --skip-replicates"
  echo "       or: python scripts/train_wedge_a_surrogate.py --config configs/wedge_a_improved.yaml"
  exit 1
fi

echo "==> Prune latent/mask caches on finished replicates (keeps meep_results.csv)"
python scripts/cleanup_sim_budget_cache.py 2>/dev/null || true

echo "==> Sim-budget replicates N=$N base_seed=$BASE_SEED"
for ((r=1; r<=N; r++)); do
  SEED=$((BASE_SEED + r * 1000))
  echo ""
  echo "======== Replicate $r / $N (seed=$SEED) ========"
  bash scripts/run_meep.sh scripts/run_sim_budget_study.py \
    --config "$CONFIG" \
    --replicate-id "$r" \
    --seed "$SEED" \
    --ensure-candidates
done

echo ""
echo "==> Aggregate"
python scripts/aggregate_sim_budget_replicates.py --config "$CONFIG"
echo "Done."
