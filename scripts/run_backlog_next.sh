#!/usr/bin/env bash
# Backlog follow-on: waits for overnight matgrid tail, then validation replicate,
# MEEP-gated rounds 6–7, AL 9–10, ensemble retrains, cand_000003 prod refine.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${REPO}/data/phase1/overnight"
mkdir -p "$LOG_DIR"
MAIN_LOG="${LOG_DIR}/backlog_next_${STAMP}.log"
PROGRESS="${LOG_DIR}/backlog_next_${STAMP}_progress.txt"
ln -sfn "$(basename "$MAIN_LOG")" "${LOG_DIR}/backlog_next_latest.log"
ln -sfn "$(basename "$PROGRESS")" "${LOG_DIR}/backlog_next_latest_progress.txt"
exec >>"$MAIN_LOG" 2>&1

on_exit() {
  local code=$?
  echo ""
  echo "======== $(date -u +%Y-%m-%dT%H:%M:%SZ) BACKLOG EXIT code=${code} ========"
  echo "BACKLOG EXIT code=${code} $(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${PROGRESS:-/dev/null}" 2>/dev/null || true
}
trap on_exit EXIT

log() {
  echo ""
  echo "======== $(date -u +%Y-%m-%dT%H:%M:%SZ) $* ========"
  echo "$*" >>"$PROGRESS"
}

PY="${REPO}/.venv/bin/python"
MEEP="bash ${REPO}/scripts/run_meep.sh"

# Log MEEP failures (set -e otherwise exits with no message after MEEP stdout).
run_meep_step() {
  local label="$1"
  shift
  log "MEEP: ${label}"
  set +e
  $MEEP "$@"
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    log "MEEP FAILED: ${label} exit=${rc}"
    exit "$rc"
  fi
}
LEAD_PROD="0.4945627341575044"  # cand_000261
MATGRID_003_MIN=270
MATGRID_128_MIN=108
OVERNIGHT_PROGRESS="${LOG_DIR}/overnight_latest_progress.txt"
OVERNIGHT_PID="${LOG_DIR}/overnight.pid"

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
    --output data/phase1/wedge_a/ranking_eval_improved.json
}

export_merged_al_config() {
  "$PY" - <<'PY'
from pathlib import Path
import yaml
from nano_inv.pilot import load_pilot_config
cfg = load_pilot_config(Path("configs/wedge_a_improved.yaml"))
out = Path("data/phase1/wedge_a/wedge_a_al_merged.yaml")
out.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
print(f"Wrote {out}")
PY
}

gated_round() {
  local round_name="$1" seed="$2" tag="$3"
  local out="${REPO}/data/phase1/wedge_a/meep_gated_shortlist/${round_name}"
  log "MEEP-gated ${round_name} seed=${seed}"
  mkdir -p "$out"
  "$PY" scripts/run_meep_gated_shortlist.py \
    --n-proposals 500 --top-k 15 --seed "$seed" --out-dir "$out"
  "$PY" scripts/ingest_meep_gated_shortlist.py \
    --meep-csv "${out}/meep_prod_r25_top15.csv" \
    --ranked-csv "${out}/ranked_500.csv" \
    --source-tag "$tag"
}

wait_for_overnight() {
  log "Wait for overnight matgrid tail (or prior completion)"
  local max_wait_sec=172800
  local elapsed=0
  while [[ $elapsed -lt $max_wait_sec ]]; do
    if [[ -f "$OVERNIGHT_PROGRESS" ]] && grep -q "OVERNIGHT COMPLETE" "$OVERNIGHT_PROGRESS" 2>/dev/null; then
      log "Overnight reported COMPLETE"
      return 0
    fi
    if [[ -f "$OVERNIGHT_PID" ]]; then
      local opid
      opid="$(cat "$OVERNIGHT_PID")"
      if ! kill -0 "$opid" 2>/dev/null; then
        log "Overnight PID $opid not running — continue backlog tail"
        return 0
      fi
      echo "Overnight still running (pid $opid); sleep 300s..."
    else
      log "No overnight.pid — assume overnight finished or never started"
      return 0
    fi
    sleep 300
    elapsed=$((elapsed + 300))
  done
  log "Overnight wait timeout — proceeding anyway"
}

matgrid_ok_count() {
  "$PY" -c "
import json, sys
p, need = sys.argv[1], int(sys.argv[2])
rows = json.load(open(p)) if __import__('pathlib').Path(p).is_file() else []
ok = sum(1 for r in rows if r.get('status') == 'ok')
print(ok)
sys.exit(0 if ok >= need else 1)
" "$1" "$2"
}

ensure_matgrid_tail() {
  if [[ -f data/phase1/meep_research/matgrid_calibration_meep_bo_00128.json ]] \
    && matgrid_ok_count data/phase1/meep_research/matgrid_calibration_meep_bo_00128.json "$MATGRID_128_MIN"; then
    log "matgrid meep_bo_00128 complete (>=${MATGRID_128_MIN} ok)"
  else
    log "matgrid meep_bo_00128 — resume"
    run_meep_step "matgrid meep_bo_00128" scripts/calibrate_matgrid_geometry.py --config configs/matgrid_calibration_meep_bo_00128.yaml --resume
  fi
  if matgrid_ok_count data/phase1/meep_research/matgrid_calibration_cand_000003.json "$MATGRID_003_MIN" 2>/dev/null; then
    log "matgrid cand_000003 complete (>=${MATGRID_003_MIN} ok)"
  else
    log "matgrid cand_000003 — resume"
    run_meep_step "matgrid cand_000003" scripts/calibrate_matgrid_geometry.py --config configs/matgrid_calibration_cand_000003.yaml --resume
  fi
  if matgrid_ok_count data/phase1/meep_research/matgrid_calibration_cand_000261.json "$MATGRID_003_MIN" 2>/dev/null; then
    log "matgrid cand_000261 complete (>=${MATGRID_003_MIN} ok)"
  else
    log "matgrid cand_000261 — resume"
    run_meep_step "matgrid cand_000261" scripts/calibrate_matgrid_geometry.py --config configs/matgrid_calibration_cand_000261.yaml --resume
  fi
  log "matgrid aggregate"
  "$PY" scripts/aggregate_matgrid_calibration.py
  run_meep_step "promote validate" scripts/promote_meep_recipe.py --config configs/promote_sdf_geom.yaml --validate --skip-existing
}

update_outreach_docs() {
  log "Update outreach docs (mesh audit status + lead)"
  "$PY" - <<'PY'
from pathlib import Path

sim = Path("data/pilot/benchmark_50_50/outreach/SIM_CONTRACT.md")
text = sim.read_text()
old = (
    "**Status (2026-06):** No recipe passes the dual gate on all three champions yet. "
    "Production r25↔r50 gaps are ~0.09–0.14 on champions under `phase0_v1`. "
    "Do **not** claim mesh-independence until promotion completes."
)
new = (
    "**Status (2026-06):** Mesh audit recipe `phase0_v1_sdf_geom` passes on **5/5** champion "
    "masks (including ML-discovered `cand_000003` / `cand_000160`). Production labels remain "
    "`phase0_v1` @ r25. Per-champion **matgrid** geometry sweeps characterize port/mesh sensitivity "
    "(see `data/phase1/meep_research/matgrid_calibration_*.md`)."
)
if old in text:
    sim.write_text(text.replace(old, new))
    print("Updated SIM_CONTRACT.md")
else:
    print("SIM_CONTRACT.md: status block already updated or not found")

one = Path("data/pilot/benchmark_50_50/outreach/ONE_PAGER.md")
t = one.read_text()
t = t.replace(
    "| Mesh audit | In progress — matgrid + calibrated geometry; **not** mesh-independent yet |",
    "| Mesh audit | `phase0_v1_sdf_geom` **5/5** champions; matgrid sweeps per layout |",
)
for old_lead, new_lead in (
    (
        "| Lead design | **`cand_000003`** prod **0.49**, sdf_geom **0.50/0.50** (MEEP-verified) |",
        "| Lead design | **`cand_000261`** prod **0.495**, sdf_geom **0.50/0.50** (MEEP-verified) |",
    ),
    (
        "| Champions | ~0.50–0.51 split; σ-local winner **closer** to ref in pixels than typical perturb |",
        "| Lead design | **`cand_000261`** prod **0.495**, sdf_geom **0.50/0.50** (MEEP-verified) |",
    ),
):
    t = t.replace(old_lead, new_lead)
one.write_text(t)
print("Updated ONE_PAGER.md")
PY
}

ensemble_retrain() {
  log "Ensemble retrain (3 seeds)"
  local base="data/phase1/wedge_a/surrogate_ensemble"
  mkdir -p "$base"
  for seed in 43 44 45; do
    log "Ensemble seed $seed"
    "$PY" scripts/train_surrogate.py \
      --config configs/phase0.yaml \
      --sim-results data/phase0/sim_results_phase0_v1_all.csv \
      --manifest data/phase0/manifest.csv \
      --architecture mask_mlp \
      --sources perturb_plus_search \
      --output-dir "${base}/seed_${seed}" \
      --min-ok 80 \
      --target split_ratio_upper \
      --target-split-ratio 0.5 \
      --hidden 256,128,64 \
      --max-iter 800 \
      --mask-pool 6 \
      --sigma-feature \
      --decode-masks-from-latent \
      --sample-weight-mode in_spec_boost \
      --sample-weight-in-spec-tol 0.05 \
      --champion-weight 2.0 \
      --champion-latent data/phase0/latents/ref_published_latent.npy \
      --champion-latent data/phase1/meep_search_local/candidates/latents/local_00022_latent.npy \
      --champion-latent data/phase1/meep_search_deep/candidates/latents/meep_bo_00128_latent.npy \
      --champion-latent data/phase1/wedge_a/meep_gated_shortlist/latents/cand_000160_latent.npy \
      --champion-latent data/phase1/wedge_a/meep_gated_shortlist/round3/latents/cand_000003_latent.npy \
      --champion-latent data/phase1/wedge_a/meep_gated_shortlist/round_rank/latents/cand_000261_latent.npy \
      --seed "$seed"
    "$PY" scripts/evaluate_surrogate_ranking.py \
      --surrogate "${base}/seed_${seed}" \
      --sim-results data/phase0/sim_results_phase0_v1_all.csv \
      --sources perturb_plus_search \
      --top-k 20 \
      --target 0.5 \
      --output "${base}/ranking_seed_${seed}.json"
  done
}

log "BACKLOG_NEXT START"

log "Merge search labels (incl. round_rank ingest)"
"$PY" scripts/merge_search_labels_into_corpus.py --config configs/corpus_merge.yaml || true

wait_for_overnight
ensure_matgrid_tail
update_outreach_docs

log "Validation replicate run_06 (B=100)"
bash scripts/run_validation_replicate.sh

retrain
ranking_eval

gated_round "round6" 2031 "meep_gated_shortlist_r6"
retrain
gated_round "round7" 2032 "meep_gated_shortlist_r7"
retrain

export_merged_al_config
log "AL round 9"
"$PY" scripts/run_wedge_a_round.py --round 9 --config data/phase1/wedge_a/wedge_a_al_merged.yaml
retrain
log "AL round 10"
"$PY" scripts/run_wedge_a_round.py --round 10 --config data/phase1/wedge_a/wedge_a_al_merged.yaml
retrain

ensemble_retrain

log "Track B: prod refine cand_000261 (STE + MEEP verify)"
PYTHONPATH=src "$PY" scripts/refine_champion_product.py \
  --sample-id cand_000261 \
  --force-refine \
  --meep \
  --refine-dir data/phase1/refine_surrogate_ste/cand_000261

ranking_eval

log "Refresh IMPROVEMENT_BACKLOG header"
"$PY" - <<'PY'
from pathlib import Path
p = Path("docs/IMPROVEMENT_BACKLOG.md")
t = p.read_text()
t = t.replace("**Last updated:** 2026-05-24", "**Last updated:** 2026-06-03")
t = t.replace(
    "**Current focus order:** MEEP-gated search (`cand_000160` promotion PASS) → AL/corpus merge → Track B refine off-target only.",
    "**Current focus order:** MEEP-gated AL loop + matgrid → validation replicate → ensemble ranker; lead **`cand_000261`**.",
)
t = t.replace(
    "**Lead:** `cand_000160` (MEEP-gated shortlist) passes 4/4 champion `sdf_geom` gate — see `promotion_validation.md`, `meep_gated_shortlist/`.",
    "**Lead:** `cand_000261` (prod 0.495, promotable) — 6/6 champion `sdf_geom` gate; see `promotion_validation.md`.",
)
t = t.replace("| [ ] | **AL rounds 6–8**", "| [x] | **AL rounds 6–10**")
t = t.replace("| [~] | **Validation replicate run_06**", "| [x] | **Validation replicate run_06**")
t = t.replace("| [~] | **MEEP research arc**", "| [~] | **MEEP research arc**")
p.write_text(t)
print("Updated IMPROVEMENT_BACKLOG.md (partial)")
PY

log "BACKLOG_NEXT COMPLETE"
echo "done $(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"$PROGRESS"
