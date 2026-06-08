#!/usr/bin/env bash
# Finish steps that failed or were skipped in run_phase1_deep / resume.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate

echo "==> A/3 MEEP re-verify deep-search champion (3× resim)"
MANIFEST=data/phase1/meep_search_deep/top_candidates.csv
for i in 1 2 3; do
  bash scripts/run_meep.sh scripts/run_fdtd_batch.py \
    --config configs/phase0.yaml \
    --manifest "$MANIFEST" \
    --sample-ids meep_bo_00128 \
    --output "data/phase1/meep_bo_00128_resim_${i}.csv" \
    --force-resim --no-skip-existing
done

echo "==> B/3 AL round summary + optional retrain (skip if already done)"
if [[ ! -f data/phase1/al_round_01/surrogate/surrogate.joblib ]]; then
  python scripts/surrogate_ranked_al_round.py --round 1 --skip-presearch --skip-meep
else
  python scripts/surrogate_ranked_al_round.py --round 1 \
    --skip-presearch --skip-meep --skip-train --skip-ranking
fi

echo "==> C/3 Ranking eval (idempotent)"
python scripts/evaluate_surrogate_ranking.py \
  --surrogate data/phase1/al_round_01/surrogate \
  --sim-results data/phase0/sim_results_phase0_v1_all.csv \
  --sources all \
  --output data/phase1/al_round_01/ranking_eval.json

echo "==> D/3 Closeout JSON"
python scripts/write_phase1_deep_closeout.py

echo "Done — see data/phase1/phase1_deep_closeout.json"
