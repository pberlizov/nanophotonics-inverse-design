#!/usr/bin/env bash
# Full Phase 0 closeout: scale MEEP search, merge labels, train surrogate, export best design.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

echo "==> 1/5 MEEP search 100 trials (~6 min)"
bash scripts/run_meep.sh scripts/meep_search.py \
  --n-trials 100 \
  --output-dir data/phase0/meep_search_100

echo "==> 2/5 MEEP-label top-10 from 100-trial search"
bash scripts/run_meep.sh scripts/run_fdtd_batch.py \
  --manifest data/phase0/meep_search_100/top_candidates.csv \
  --force-resim --no-skip-existing

echo "==> 3/5 Merge corpus + train mask_mlp surrogate"
source .venv/bin/activate
python scripts/finalize_phase0.py --best-id meep_bo_00010

BEST_ID="$(python - <<'PY' 2>/dev/null || echo meep_bo_00010
import json
from pathlib import Path
import pandas as pd
root = Path("data/phase0/meep_search_100")
if not (root / "meep_search_summary.json").exists():
    print("meep_bo_00010")
    raise SystemExit
s = json.loads((root / "meep_search_summary.json").read_text())
t = pd.read_csv(root / "meep_search_trials.csv")
print(t.loc[t.trial_number == s["best_trial"], "sample_id"].iloc[0])
PY
)"

echo "==> 4/5 Export best design PNG ($BEST_ID)"
python scripts/export_best_design.py --sample-id "$BEST_ID" || true

echo "==> 5/5 Gate metrics"
python scripts/evaluate_phase0.py --output data/phase0/gate_metrics_final.json

echo "Done. See data/phase0/phase0_closeout.json and docs/phase0_results.md"
