#!/usr/bin/env bash
# Production prep: merge → AL rounds → train frozen surrogate → (optional) 5× sim-budget replicates.
#
#   bash scripts/run_production_pipeline.sh
#   bash scripts/run_production_pipeline.sh --skip-replicates
#   bash scripts/run_production_pipeline.sh --skip-al --only-replicates
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
source .venv/bin/activate

SKIP_AL=0
SKIP_REPLICATES=0
ONLY_REPLICATES=0
SKIP_MERGE=0

for arg in "$@"; do
  case "$arg" in
    --skip-al) SKIP_AL=1 ;;
    --skip-replicates) SKIP_REPLICATES=1 ;;
    --only-replicates) ONLY_REPLICATES=1; SKIP_AL=1; SKIP_MERGE=1 ;;
    --skip-merge) SKIP_MERGE=1 ;;
  esac
done

PROD_CFG="configs/wedge_a_production.yaml"

if [[ "$ONLY_REPLICATES" != 1 ]]; then
  if [[ "$SKIP_MERGE" != 1 ]]; then
    echo "==> 1/5 Merge search labels into corpus"
    python scripts/merge_search_labels_into_corpus.py --config configs/corpus_merge.yaml
  fi

  if [[ "$SKIP_AL" != 1 ]]; then
    echo "==> 2/5 Active learning rounds 2–5 (MEEP + retrain between rounds)"
    python scripts/run_wedge_a_al_batch.py --config "$PROD_CFG" --from-round 2 --to-round 5
  fi

  echo "==> 3/5 Train production surrogate (frozen for replicates)"
  python scripts/train_wedge_a_surrogate.py --config "$PROD_CFG"

  echo "==> 4/5 Ranking gate on production surrogate"
  python scripts/evaluate_surrogate_ranking.py \
    --surrogate data/phase1/wedge_a/surrogate_improved \
    --sim-results data/phase0/sim_results_phase0_v1_all.csv \
    --sources perturb_plus_search \
    --output data/phase1/wedge_a/ranking_eval_production.json

  echo "==> 4b/5 Optional: abs_split_error production model"
  python scripts/train_wedge_a_surrogate.py --config configs/wedge_a_mask_abserr.yaml \
    --skip-ranking 2>/dev/null || \
  python scripts/train_wedge_a_surrogate.py --config configs/wedge_a_mask_abserr.yaml || true
fi

if [[ "$SKIP_REPLICATES" != 1 ]]; then
  echo "==> 5/5 Sim-budget replicates (5 seeds × policies × B=30,50,100)"
  echo "    This is ~weekend MEEP compute. Requires conda env mp."
  bash scripts/run_sim_budget_replicates.sh
  bash scripts/run_pilot.sh --config configs/pilot/benchmark_50_50.yaml --report-only
else
  echo "==> 5/5 Skipped replicates (--skip-replicates)"
fi

echo ""
echo "Production pipeline done."
echo "  Surrogate: data/phase1/wedge_a/surrogate_improved"
echo "  Replicates: data/phase1/wedge_a/sim_budget_replication_report.md"
