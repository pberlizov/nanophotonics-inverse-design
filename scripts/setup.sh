#!/usr/bin/env bash
# Create .venv with Python 3.12.12 and install drcgenerator.
# Run from repo root:  bash scripts/setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="${HOME}/.local/bin:${PATH}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "==> Removing old .venv (fixes 3.11 / 3.14 mistakes)"
rm -rf .venv

echo "==> Installing CPython 3.12.12"
uv python install 3.12.12

echo "==> Creating .venv"
uv venv --python 3.12.12 .venv

PY="${REPO_ROOT}/.venv/bin/python"
echo "==> Python: $("$PY" --version) at $PY"

mkdir -p external
if [[ ! -d external/drcgenerator ]]; then
  echo "==> Cloning drcgenerator"
  git clone --depth 1 https://github.com/Photonic-Architecture-Laboratories/drcgenerator.git external/drcgenerator
fi

echo "==> Installing drcgenerator (use uv pip — do not use system pip)"
uv pip install -e external/drcgenerator --python "$PY"

echo "==> Verify"
"$PY" -c "import jax, drcgenerator; print('OK: jax', jax.__version__, 'drcgenerator', drcgenerator.__file__)"

echo ""
echo "Done. Activate with:"
echo "  source ${REPO_ROOT}/.venv/bin/activate"
echo "Then confirm:  which python   # must end with nanophotonics-inverse-design/.venv/bin/python"
echo "               python --version   # must be Python 3.12.12"
