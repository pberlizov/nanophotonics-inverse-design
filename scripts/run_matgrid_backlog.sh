#!/usr/bin/env bash
# Run per-champion matgrid geometry sweeps (backlog 2026-06).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

run_one() {
  local cfg="$1"
  echo "==> $cfg"
  bash scripts/run_meep.sh scripts/calibrate_matgrid_geometry.py --config "$cfg" --resume
}

case "${1:-all}" in
  local_00022)
    run_one configs/matgrid_calibration_local_00022.yaml
    ;;
  meep_bo_00093)
    run_one configs/matgrid_calibration_meep_bo_00093.yaml
    ;;
  meep_bo_00128)
    run_one configs/matgrid_calibration_meep_bo_00128.yaml
    ;;
  all)
    run_one configs/matgrid_calibration_local_00022.yaml
    run_one configs/matgrid_calibration_meep_bo_00093.yaml
    PYTHONPATH=src python scripts/aggregate_matgrid_calibration.py
    ;;
  *)
    echo "Usage: $0 [local_00022|meep_bo_00093|meep_bo_00128|all]"
    exit 1
    ;;
esac

PYTHONPATH=src python scripts/aggregate_matgrid_calibration.py
