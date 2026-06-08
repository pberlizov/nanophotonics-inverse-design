#!/usr/bin/env bash
# Resume Track B after B1 residual completed (skips B2 + residual search).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate

PILOT=1
[[ "${1:-}" == "--full" ]] && PILOT=0

CFG=configs/phase1_track_b.yaml
MEEP=(bash scripts/run_meep.sh)
N_LATENT=20
N_TPL=15
N_ACQ=200
[[ "$PILOT" == 0 ]] && { N_LATENT=80; N_TPL=50; N_ACQ=1000; }

echo "==> B1 PCA (basis from .venv)"
python scripts/fit_latent_pca_basis.py --config "$CFG"
"${MEEP[@]}" scripts/latent_meep_search.py --config "$CFG" \
  --latent-mode pca --n-trials "$((N_LATENT / 2))" \
  --output-dir data/phase1/track_b/latent_meep_pca

echo "==> B3 Template search"
"${MEEP[@]}" scripts/meep_template_search.py --config "$CFG" --n-trials "$N_TPL"

echo "==> B5 Acquisition"
python scripts/surrogate_acquisition_search.py --config "$CFG" \
  --surrogate data/phase1/track_b/surrogates/perturb_latent_mlp \
  --n-proposals "$N_ACQ" --top-k 25

echo "==> B7 Baselines"
python scripts/run_baselines.py --config "$CFG" $([[ "$PILOT" == 1 ]] && echo --pilot)

echo "==> B4 Pareto"
python scripts/summarize_pareto_trials.py \
  --trials data/phase1/track_b/latent_meep_search/latent_meep_trials.csv \
  --output data/phase1/track_b/pareto_latent_meep.json || true

echo "Done resume."
