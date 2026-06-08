#!/usr/bin/env bash
# R² improvement pipeline: merge search labels → train variants → compare metrics → optional AL.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
source .venv/bin/activate

RUN_AL="${RUN_AL:-0}"
DRY_MERGE="${DRY_MERGE:-0}"

echo "==> 1/4 Merge MEEP search + champion labels into corpus"
if [[ "$DRY_MERGE" == 1 ]]; then
  python scripts/merge_search_labels_into_corpus.py --dry-run
else
  python scripts/merge_search_labels_into_corpus.py --config configs/corpus_merge.yaml
fi

echo "==> 2/4 Train surrogate variants"
for cfg in configs/wedge_a.yaml configs/wedge_a_mask.yaml configs/wedge_a_mask_abserr.yaml; do
  echo "    --- $cfg"
  python scripts/train_wedge_a_surrogate.py --config "$cfg"
done

echo "==> 3/4 Comparison table (ranking eval only — training done in step 2)"
python scripts/compare_surrogate_variants.py

if [[ "$RUN_AL" == 1 ]]; then
  echo "==> 4/4 Active learning rounds 2-4 (MEEP)"
  python scripts/run_wedge_a_al_batch.py --config configs/wedge_a_mask.yaml --from-round 2 --to-round 4
else
  echo "==> 4/4 Skip AL (set RUN_AL=1 to run rounds 2-4 with MEEP)"
fi

echo "Done. See data/phase1/surrogate_variant_comparison.csv"
