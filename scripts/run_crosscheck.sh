#!/usr/bin/env bash
# Run full multi-solver champion cross-check (MEEP in mp env, Tidy3D/BPM in .venv).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> MEEP solvers (mp env)"
bash scripts/run_meep.sh scripts/crosscheck_champion_solvers.py --solvers meep

echo "==> Tidy3D cloud (optional; skips if no API key)"
PYTHONPATH=src .venv/bin/python scripts/crosscheck_champion_solvers.py --solvers tidy3d --append || true

echo "==> Done: data/phase1/crosscheck/"
