#!/usr/bin/env bash
# Run any script with the MEEP (mp) environment via micromamba — no conda activate needed.
# Example:
#   bash scripts/run_meep.sh scripts/run_fdtd_batch.py --limit 10 --resolution 15

set -euo pipefail

MFORGE="${HOME}/miniforge3"
export MAMBA_ROOT_PREFIX="$MFORGE"

if [[ ! -x "$MFORGE/micromamba" ]]; then
  echo "ERROR: $MFORGE/micromamba not found. Run: bash scripts/install_meep.sh"
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_meep.sh <python-script> [args...]"
  exit 1
fi

if ! "$MFORGE/micromamba" env list | awk '{print $1}' | grep -qx mp; then
  echo "ERROR: env 'mp' not found. Run: bash scripts/install_meep.sh"
  exit 1
fi

# One-time style: ensure batch/search deps exist in mp (not bundled with pymeep)
"$MFORGE/micromamba" run -n mp python -c "import pandas, optuna, tqdm, yaml, joblib, sklearn, noise" 2>/dev/null || {
  echo "==> Installing MEEP-side deps into env mp (pandas, optuna, joblib, sklearn, …)"
  "$MFORGE/micromamba" run -n mp python -m pip install -q pandas pyyaml tqdm numpy optuna joblib scikit-learn scipy noise
}

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}/src"
exec "$MFORGE/micromamba" run -n mp python "$@"
