#!/usr/bin/env bash
# Track B — start all structural improvement pilots (fast) or full runs.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate

PILOT=1
if [[ "${1:-}" == "--full" ]]; then
  PILOT=0
  shift
fi

CFG=configs/phase1_track_b.yaml
MEEP=(bash scripts/run_meep.sh)

echo "==> B2 Train split surrogates (~1 min)"
python scripts/train_track_b_surrogates.py --config "$CFG"

if [[ "$PILOT" == 1 ]]; then
  N_LATENT=20
  N_TPL=15
  N_SIGMA=15
  N_ACQ=200
  echo "==> PILOT mode (pass --full for larger searches)"
else
  N_LATENT=80
  N_TPL=50
  N_SIGMA=100
  N_ACQ=1000
fi

echo "==> B1 Latent MEEP search (residual)"
"${MEEP[@]}" scripts/latent_meep_search.py --config "$CFG" \
  --latent-mode residual --n-trials "$N_LATENT"

echo "==> B1 Latent MEEP search (pca)"
python scripts/fit_latent_pca_basis.py --config "$CFG"
"${MEEP[@]}" scripts/latent_meep_search.py --config "$CFG" \
  --latent-mode pca --n-trials "$((N_LATENT / 2))" \
  --output-dir data/phase1/track_b/latent_meep_pca

echo "==> B3 Template + mask co-search"
"${MEEP[@]}" scripts/meep_template_search.py --config "$CFG" --n-trials "$N_TPL"

echo "==> B5 Surrogate acquisition presearch"
python scripts/surrogate_acquisition_search.py \
  --config "$CFG" \
  --surrogate data/phase1/track_b/surrogates/perturb_latent_mlp \
  --n-proposals "$N_ACQ" --top-k 25

echo "==> B7 Baselines comparison"
python scripts/run_baselines.py --config "$CFG" $([[ "$PILOT" == 1 ]] && echo --pilot)

echo "==> B4 Pareto summary from latent search trials"
python scripts/summarize_pareto_trials.py \
  --trials data/phase1/track_b/latent_meep_search/latent_meep_trials.csv \
  --output data/phase1/track_b/pareto_latent_meep.json || true

echo "Done Track B. See docs/PHASE1_TRACK_B.md and data/phase1/track_b/"
