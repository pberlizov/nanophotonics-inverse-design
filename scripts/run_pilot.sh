#!/usr/bin/env bash
# Pilot pipeline: spec YAML → train → (optional MEEP) → deliverables → outreach bundle.
#
#   bash scripts/run_pilot.sh --config configs/pilot/benchmark_50_50.yaml
#   bash scripts/run_pilot.sh --config configs/pilot/benchmark_50_50.yaml --report-only
#   bash scripts/run_pilot.sh --config configs/pilot/benchmark_50_50.yaml --full-meep
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

CONFIG="${REPO}/configs/pilot/benchmark_50_50.yaml"
REPORT_ONLY=0
FULL_MEEP=0
SKIP_TRAIN=0
RETRAIN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --report-only) REPORT_ONLY=1; shift ;;
    --full-meep) FULL_MEEP=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    --retrain) RETRAIN=1; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

source .venv/bin/activate
PY=.venv/bin/python

echo "==> Pilot config: $CONFIG"

if [[ "$REPORT_ONLY" != 1 ]]; then
  echo "==> 0/5 Verify environment"
  "$PY" scripts/verify_setup.py

  if [[ "$SKIP_TRAIN" != 1 ]]; then
    echo "==> 1/5 Train wedge-A ranker"
    if [[ "$RETRAIN" == 1 ]]; then
      "$PY" scripts/train_wedge_a_surrogate.py --config "$CONFIG"
    else
      SUR_DIR=$("$PY" -c "
import yaml
from pathlib import Path
from nano_inv.pilot import load_pilot_config
cfg = load_pilot_config(Path('$CONFIG'))
print(cfg['surrogate']['output_dir'])
")
      if [[ ! -f "${SUR_DIR}/surrogate.joblib" ]]; then
        "$PY" scripts/train_wedge_a_surrogate.py --config "$CONFIG"
      else
        echo "    (surrogate exists — skip; use --retrain to force)"
      fi
    fi
    RANK_OUT=$("$PY" -c "
from pathlib import Path
from nano_inv.pilot import load_pilot_config, resolve_path
cfg = load_pilot_config(Path('$CONFIG'))
print(resolve_path(cfg.get('sources', {}).get('ranking_eval', 'data/phase1/wedge_a/ranking_eval.json')))
")
    "$PY" scripts/evaluate_surrogate_ranking.py \
      --surrogate data/phase1/wedge_a/surrogate \
      --sim-results data/phase0/sim_results_phase0_v1_all.csv \
      --sources perturb \
      --output "$RANK_OUT" || true
  fi

  echo "==> 2/5 Sim-budget study (MEEP)"
  MEEP_ARGS=(scripts/run_sim_budget_study.py --config "$CONFIG" --ensure-candidates)
  if [[ "$FULL_MEEP" == 1 ]]; then
    bash scripts/run_meep.sh "${MEEP_ARGS[@]}"
  else
    bash scripts/run_meep.sh "${MEEP_ARGS[@]}" --pilot
  fi
else
  echo "==> Report-only mode (skip train + MEEP)"
fi

echo "==> 3/6 Characterize design novelty vs ref_published"
"$PY" scripts/characterize_design_novelty.py

echo "==> 4/6 Build design dossier"
"$PY" scripts/build_pilot_deliverables.py --config "$CONFIG"

echo "==> 5/6 Generate sim-budget report + plot"
"$PY" scripts/generate_pilot_report.py --config "$CONFIG"
"$PY" scripts/render_sim_budget_plot.py --config "$CONFIG"

echo "==> 6/6 Package outreach bundle"
"$PY" scripts/package_pilot_outreach.py --config "$CONFIG"

DLV=$("$PY" -c "
from pathlib import Path
from nano_inv.pilot import load_pilot_config, deliverables_dir
print(deliverables_dir(load_pilot_config(Path('$CONFIG'))))
")
OUTREACH=$("$PY" -c "
from pathlib import Path
from nano_inv.pilot import load_pilot_config, resolve_path
cfg = load_pilot_config(Path('$CONFIG'))
print(resolve_path(cfg.get('outreach', {}).get('dir', 'data/pilot/' + cfg['pilot']['id'] + '/outreach')))
")

echo ""
echo "Done."
echo "  Deliverables: $DLV"
echo "  Outreach:     $OUTREACH"
echo "  Next: review ONE_PAGER.md, fill contact_email, run --full-meep before external claims."
