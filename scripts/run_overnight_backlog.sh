#!/usr/bin/env bash
# Overnight sequential backlog (Track A heavy MEEP + matgrid).
# Started manually; logs under data/phase1/overnight/
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${REPO}/data/phase1/overnight"
mkdir -p "$LOG_DIR"
MAIN_LOG="${LOG_DIR}/overnight_${STAMP}.log"
PROGRESS="${LOG_DIR}/overnight_${STAMP}_progress.txt"
ln -sfn "$(basename "$MAIN_LOG")" "${LOG_DIR}/overnight_latest.log"
ln -sfn "$(basename "$PROGRESS")" "${LOG_DIR}/overnight_latest_progress.txt"
exec >>"$MAIN_LOG" 2>&1

log() {
  echo ""
  echo "======== $(date -u +%Y-%m-%dT%H:%M:%SZ) $* ========"
  echo "$*" >>"$PROGRESS"
}

PY="${REPO}/.venv/bin/python"
MEEP="bash ${REPO}/scripts/run_meep.sh"
LEAD_PROD="0.4902372917217212"  # cand_000003

retrain() {
  log "Retrain surrogate_improved"
  "$PY" scripts/train_wedge_a_surrogate.py --config configs/wedge_a_improved.yaml
}

ranking_eval() {
  log "Ranking eval"
  "$PY" scripts/evaluate_surrogate_ranking.py \
    --surrogate data/phase1/wedge_a/surrogate_improved \
    --sim-results data/phase0/sim_results_phase0_v1_all.csv \
    --sources perturb_plus_search \
    --top-k 20 \
    --target 0.5 \
    --output data/phase1/wedge_a/ranking_eval_improved.json || true
}

export_merged_al_config() {
  log "Export merged AL config"
  "$PY" - <<'PY'
from pathlib import Path
import yaml
from nano_inv.pilot import load_pilot_config
cfg = load_pilot_config(Path("configs/wedge_a_improved.yaml"))
out = Path("data/phase1/wedge_a/wedge_a_al_merged.yaml")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
print(f"Wrote {out}")
PY
}

gated_round() {
  local round_name="$1"
  local seed="$2"
  local tag="$3"
  local out="${REPO}/data/phase1/wedge_a/meep_gated_shortlist/${round_name}"
  log "MEEP-gated shortlist ${round_name} seed=${seed}"
  mkdir -p "$out"
  "$PY" scripts/run_meep_gated_shortlist.py \
    --n-proposals 500 --top-k 15 --seed "$seed" --out-dir "$out"
  log "Ingest ${tag}"
  "$PY" scripts/ingest_meep_gated_shortlist.py \
    --meep-csv "${out}/meep_prod_r25_top15.csv" \
    --ranked-csv "${out}/ranked_500.csv" \
    --source-tag "$tag"
  check_gated_winner "${out}/meep_gated_summary.json" "${round_name}"
}

check_gated_winner() {
  local summary="$1"
  local label="$2"
  log "Check promotion candidate from ${label}"
  "$PY" - <<PY
import json
from pathlib import Path
p = Path("$summary")
if not p.exists():
    print("No summary:", p)
    raise SystemExit(0)
data = json.loads(p.read_text())
best = data.get("best") or []
if not best:
    print("No best entries")
    raise SystemExit(0)
row = best[0]
sid = row["sample_id"]
prod = float(row["split_ratio_upper"])
err = float(row["abs_err"])
lead = float("$LEAD_PROD")
print(f"Best {sid} prod={prod:.4f} |err|={err:.4f} vs lead {lead:.4f}")
if err <= 0.02 and prod >= lead - 1e-6:
    import subprocess, os
    repo = Path("$REPO")
    mask = repo / "data/phase1/wedge_a/meep_gated_shortlist" / Path("$label") / "masks" / f"{sid}_mask.npy"
    if not mask.exists():
        for base in [repo / "data/phase1/wedge_a/meep_gated_shortlist"]:
            alt = base / "masks" / f"{sid}_mask.npy"
            if alt.exists():
                mask = alt
                break
    if mask.exists():
        subprocess.run([
            "bash", str(repo / "scripts/run_meep.sh"),
            str(repo / "scripts/verify_promotion_candidate.py"),
            "--candidate-mask", str(mask),
            "--candidate-id", sid,
            "--compare-id", "local_00022",
        ], cwd=repo, check=False)
    else:
        print("Mask not found for", sid)
PY
}

al_round() {
  local n="$1"
  log "Wedge-A AL round ${n}"
  export_merged_al_config
  "$PY" scripts/run_wedge_a_round.py \
    --round "$n" \
    --config data/phase1/wedge_a/wedge_a_al_merged.yaml
}

log "OVERNIGHT START repo=$REPO"

log "Phase 0: promote recipe --apply (docs + phase0.yaml)"
$MEEP scripts/promote_meep_recipe.py --config configs/promote_sdf_geom.yaml --apply

log "Phase 1: initial retrain + ranking"
retrain
ranking_eval

gated_round "round4" 2029 "meep_gated_shortlist_r4"
retrain

gated_round "round5" 2030 "meep_gated_shortlist_r5"
retrain

al_round 7
retrain

al_round 8
retrain
ranking_eval

log "Phase matgrid: local_00022"
$MEEP scripts/calibrate_matgrid_geometry.py --config configs/matgrid_calibration_local_00022.yaml --resume

log "Phase matgrid: meep_bo_00093"
$MEEP scripts/calibrate_matgrid_geometry.py --config configs/matgrid_calibration_meep_bo_00093.yaml --resume

log "Phase matgrid: meep_bo_00128"
$MEEP scripts/calibrate_matgrid_geometry.py --config configs/matgrid_calibration_meep_bo_00128.yaml --resume

log "Phase matgrid: cand_000003"
$MEEP scripts/calibrate_matgrid_geometry.py --config configs/matgrid_calibration_cand_000003.yaml --resume

log "Phase matgrid: aggregate"
"$PY" scripts/aggregate_matgrid_calibration.py

log "Phase final: champion validate --skip-existing"
$MEEP scripts/promote_meep_recipe.py --config configs/promote_sdf_geom.yaml --validate --skip-existing

log "OVERNIGHT COMPLETE"
echo "done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$PROGRESS"
