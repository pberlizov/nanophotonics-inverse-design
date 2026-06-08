#!/usr/bin/env bash
# Post-replication improvements: merge sim-budget MEEP → retrain improved ranker → ranking gate.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
source .venv/bin/activate

echo "==> 1/3 Merge replicate MEEP labels into corpus (may decode masks — minutes)"
python scripts/merge_search_labels_into_corpus.py --config configs/corpus_merge.yaml

echo "==> 2/3 Train improved ranker (merged corpus + sigma feature)"
python scripts/train_wedge_a_surrogate.py --config configs/wedge_a_improved.yaml

echo "==> 3/3 Ranking evaluation"
python scripts/evaluate_surrogate_ranking.py \
  --surrogate data/phase1/wedge_a/surrogate_improved \
  --sim-results data/phase0/sim_results_phase0_v1_all.csv \
  --sources perturb_plus_search \
  --output data/phase1/wedge_a/ranking_eval_improved.json

echo ""
echo "Done. See:"
echo "  data/phase1/wedge_a/surrogate_improved/metrics.json"
echo "  data/phase1/wedge_a/ranking_eval_improved.json"
echo "  docs/IMPROVEMENT_BACKLOG.md"
