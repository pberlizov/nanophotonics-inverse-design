#!/usr/bin/env bash
# Install Miniforge + pymeep env "mp", or recover a broken partial install.
# Run:  bash scripts/install_meep.sh

set -euo pipefail

MFORGE="${HOME}/miniforge3"
ARCH="$(uname -m)"
if [[ "$ARCH" == "arm64" ]]; then
  INSTALLER="Miniforge3-MacOSX-arm64.sh"
elif [[ "$ARCH" == "x86_64" ]]; then
  INSTALLER="Miniforge3-MacOSX-x86_64.sh"
else
  echo "Unsupported arch: $ARCH"; exit 1
fi

micromamba() { "${MFORGE}/micromamba" "$@"; }

# --- Broken partial install: folder exists but conda never finished ---
if [[ -d "$MFORGE" && ! -f "$MFORGE/etc/profile.d/conda.sh" ]]; then
  echo "==> Found incomplete Miniforge at $MFORGE (no conda.sh)"
  if [[ -x "$MFORGE/micromamba" ]]; then
    echo "    micromamba is present — will use it to create env 'mp'"
  else
    echo "    Removing broken directory and reinstalling..."
    rm -rf "$MFORGE"
  fi
fi

# --- Full Miniforge install if missing ---
if [[ ! -x "$MFORGE/micromamba" && ! -x "$MFORGE/bin/conda" ]]; then
  echo "==> Downloading Miniforge ($INSTALLER)"
  TMP="$(mktemp -d)"
  curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER}" -o "$TMP/install.sh"
  echo "==> Installing to $MFORGE"
  bash "$TMP/install.sh" -b -p "$MFORGE"
  rm -rf "$TMP"
elif [[ -d "$MFORGE" && ! -f "$MFORGE/etc/profile.d/conda.sh" && -x "$MFORGE/micromamba" ]]; then
  echo "==> Retrying Miniforge installer with -u (complete partial install)"
  TMP="$(mktemp -d)"
  curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/${INSTALLER}" -o "$TMP/install.sh"
  bash "$TMP/install.sh" -u -b -p "$MFORGE" || true
  rm -rf "$TMP"
fi

export MAMBA_ROOT_PREFIX="$MFORGE"

# Prefer conda if available, else micromamba
if [[ -f "$MFORGE/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$MFORGE/etc/profile.d/conda.sh"
  echo "==> Creating/updating conda env 'mp'"
  if conda env list | awk '{print $1}' | grep -qx mp; then
    conda install -n mp -c conda-forge pymeep python=3.11 -y
  else
    conda create -n mp -c conda-forge pymeep python=3.11 -y
  fi
  PY="$MFORGE/envs/mp/bin/python"
else
  echo "==> Using micromamba (conda profile not found)"
  if micromamba env list | awk '{print $1}' | grep -qx mp; then
    echo "    env mp exists — installing pymeep into it"
    micromamba install -n mp -c conda-forge pymeep python=3.11 -y
  else
    micromamba create -n mp -c conda-forge pymeep python=3.11 -y
  fi
  PY="$MFORGE/envs/mp/bin/python"
fi

echo "==> Phase 0 Python deps (pandas, etc.)"
"$PY" -m pip install -q pandas pyyaml tqdm numpy optuna

echo "==> Verify meep"
"$PY" -c "import meep as mp; print('meep OK')"

echo ""
echo "Success."
echo ""
echo "Option A — conda (if conda.sh exists):"
echo "  source \"$MFORGE/etc/profile.d/conda.sh\""
echo "  conda activate mp"
echo ""
echo "Option B — micromamba (always works if micromamba exists):"
echo "  export MAMBA_ROOT_PREFIX=\"$MFORGE\""
echo "  eval \"\$(\"$MFORGE/micromamba\" shell hook -s zsh)\""
echo "  micromamba activate mp"
echo ""
echo "Option C — wrapper (no activate):"
echo "  bash scripts/run_meep.sh scripts/run_fdtd_batch.py --limit 10 --resolution 15"
