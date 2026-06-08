#!/usr/bin/env bash
# Regenerate release tables/figures under frozen phase0_v1 contract.
# For v1 preprint CPU path, prefer: bash scripts/finalize_preprint_v1.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

RUN_MEEP="${RUN_MEEP:-0}"

python scripts/build_repro_manifest.py
python scripts/characterize_design_novelty.py --extended
python scripts/surrogate_validation_regimes.py
python scripts/ablation_proposal_pool.py

python scripts/aggregate_sim_budget_replicates.py \
  --config configs/wedge_a_release_replication.yaml 2>/dev/null \
  || python scripts/aggregate_sim_budget_replicates.py \
  --config configs/wedge_a_production.yaml

if [[ "$RUN_MEEP" == "1" ]]; then
  MEEP=scripts/run_meep.sh
  bash "$MEEP" scripts/champion_fom_table.py
  bash "$MEEP" scripts/flux_il_audit.py
  bash "$MEEP" scripts/champion_broadband_sweep.py --wl-step 0.005 --max-worst-split-error 0.05
  bash "$MEEP" scripts/champion_mesh_convergence.py
  bash "$MEEP" scripts/champion_fab_stress.py
fi

bash scripts/finalize_preprint_v1.sh
echo "Release artifacts under data/phase1/release/ and docs/preprint/figures/"
