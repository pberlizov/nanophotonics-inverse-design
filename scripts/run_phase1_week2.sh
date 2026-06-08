#!/usr/bin/env bash
# Phase 1 week 2: full v1 relabel, local search, surrogate ranking eval.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

echo "==> 1a/4 Perturb cohort relabel @ phase0_v1 (~2 min) — skip if sim_results_phase0_v1.csv is current"
bash scripts/run_meep.sh scripts/relabel_recipe.py --sources perturb

echo "==> 1b/4 Full manifest relabel @ phase0_v1 (~15 min) — for surrogate on 500 masks"
bash scripts/run_meep.sh scripts/run_fdtd_batch.py \
  --recipe-version phase0_v1 \
  --output data/phase0/sim_results_phase0_v1_all.csv \
  --sources-filter all \
  --force-resim --no-skip-existing

echo "==> 2/4 Train surrogate on full v1 corpus"
source .venv/bin/activate
python scripts/train_surrogate.py \
  --sim-results data/phase0/sim_results_phase0_v1_all.csv \
  --manifest data/phase0/manifest.csv \
  --sources all \
  --architecture mask_mlp \
  --output-dir data/phase1/surrogate_mask_v1_full

echo "==> 3/4 Ranking evaluation (does surrogate beat random on MEEP?)"
python scripts/evaluate_surrogate_ranking.py \
  --surrogate data/phase1/surrogate_mask_v1_full \
  --sim-results data/phase0/sim_results_phase0_v1_all.csv \
  --sources all \
  --output data/phase1/surrogate_ranking_eval.json

echo "==> 4/4 Local MEEP search around champion sigma (~3 min)"
bash scripts/run_meep.sh scripts/meep_search_local.py --n-trials 50

echo "Done. See data/phase1/ and docs/phase1_results.md"
