#!/usr/bin/env bash
# Release backlog driver — documents order; does NOT run MEEP by default.
#
# Usage:
#   bash scripts/run_release_backlog.sh              # CPU-only steps only
#   RUN_MEEP=1 bash scripts/run_release_backlog.sh # include MEEP-heavy steps
#
# Individual MEEP scripts can also be run via:
#   bash scripts/run_meep.sh scripts/champion_fom_table.py
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-.venv/bin/python}"
RUN_MEEP="${RUN_MEEP:-0}"

step() {
  echo ""
  echo "==> $*"
}

run_cpu() {
  step "[CPU] $1"
  # shellcheck disable=SC2086
  $PY $2
}

run_meep() {
  if [[ "$RUN_MEEP" != "1" ]]; then
    step "[MEEP — skipped] $1  (set RUN_MEEP=1 to run)"
    return 0
  fi
  step "[MEEP] $1"
  bash scripts/run_meep.sh "$2"
}

echo "Release backlog @ $REPO"
echo "RUN_MEEP=$RUN_MEEP (default 0: skip MEEP simulations)"

# --- Tier A: repro + champion characterization ---
run_cpu "Reproducibility manifest" scripts/build_repro_manifest.py

run_meep "Champion FOM table (split, T, IL)" scripts/champion_fom_table.py
run_meep "Champion broadband sweep (fine grid + gate)" \
  "scripts/champion_broadband_sweep.py --wl-step 0.005 --max-worst-split-error 0.05"
run_meep "Champion mesh convergence" scripts/champion_mesh_convergence.py

# --- Budget replication (long-running; separate driver) ---
step "[MEEP — manual] Sim-budget n=20 replication"
echo "    N_REPLICATES=20 bash scripts/run_release_replication.sh"
echo "    then: $PY scripts/aggregate_sim_budget_replicates.py"

# --- Tier B: backlog scripts (this PR) ---
run_meep "Fabrication stress (±10/20/30 nm morphology)" scripts/champion_fab_stress.py
run_cpu "Surrogate validation regimes (random / group / chronological)" scripts/surrogate_validation_regimes.py
run_cpu "Proposal-pool selection ablation" scripts/ablation_proposal_pool.py
run_cpu "Extended novelty panels + NN Hamming" "scripts/characterize_design_novelty.py --extended"

# --- Optional flags (documented) ---
step "[Optional] Fab stress dry-run (list variants only)"
echo "    $PY scripts/champion_fab_stress.py --dry-run"

step "[Optional] Ablation fast mode (500-row pool)"
echo "    $PY scripts/ablation_proposal_pool.py --fast"

step "[Optional] Ablation with fresh MSE model"
echo "    $PY scripts/ablation_proposal_pool.py --train-mse"

step "[MEEP — manual] Broadband contribution hunt (winners for paper)"
echo "    bash scripts/run_broadband_hunt.sh"
echo "    pilot: TRIALS_PER_CENTER=10 SKIP_EXPLORE=1 bash scripts/run_broadband_hunt.sh"

step "[Optional] Broadband local search (legacy single-stage)"
echo "    N_TRIALS=30 bash scripts/run_broadband_search.sh"

step "[Optional] Rescore existing local candidates on fine C-band grid"
echo "    bash scripts/run_meep.sh scripts/broadband_rescore_candidates.py --wl-step 0.005"

step "[Optional] Regenerate all release artifacts"
echo "    bash scripts/regenerate_release_artifacts.sh  (if present)"

echo ""
echo "Done. Outputs under data/phase1/release/"
