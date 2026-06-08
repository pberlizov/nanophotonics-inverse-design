#!/usr/bin/env bash
# Release replication: default N=20, B=30,50,100,200, frozen surrogate_improved.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export N_REPLICATES="${N_REPLICATES:-20}"
export CONFIG="${CONFIG:-configs/wedge_a_release_replication.yaml}"
export BASE_SEED="${BASE_SEED:-2026}"
echo "Release replication: N=$N_REPLICATES config=$CONFIG"
exec bash scripts/run_sim_budget_replicates.sh
