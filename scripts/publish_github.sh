#!/usr/bin/env bash
# One-shot: init git (if needed), commit, create public GitHub repo, push, tag v1.0-preprint.
#
# Prereq: gh auth login   (token was invalid in CI/agent environment)
#
# Usage:
#   cd ~/nanophotonics-inverse-design
#   bash scripts/publish_github.sh
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
REMOTE="https://github.com/pberlizov/nanophotonics-inverse-design.git"

if ! gh auth status >/dev/null 2>&1; then
  echo "Run: gh auth login"
  exit 1
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  rm -rf .git
  git init
  git branch -M main
fi

git add -A
if git diff --cached --quiet; then
  echo "Nothing to commit."
else
  git commit -m "$(cat <<'EOF'
Open-source v1.0-preprint release.

MEEP-gated search benchmark for DRC-feasible 1×2 splitters: manuscript,
figures, OSS docs, and Zenodo bundle script. Simulation-only scope.
EOF
)"
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "$REMOTE"
fi

if gh repo view pberlizov/nanophotonics-inverse-design >/dev/null 2>&1; then
  echo "Repo exists — pushing to origin main"
  git push -u origin main
else
  gh repo create pberlizov/nanophotonics-inverse-design --public --source=. --remote=origin --push
fi

gh repo edit pberlizov/nanophotonics-inverse-design --visibility public 2>/dev/null || true

if ! git rev-parse v1.0-preprint >/dev/null 2>&1; then
  git tag -a v1.0-preprint -m "Preprint v1.0 — simulation-only research release"
  git push origin v1.0-preprint
fi

echo ""
echo "Public repo: https://github.com/pberlizov/nanophotonics-inverse-design"
echo "Zenodo zip:  data/phase1/release/nanophotonics_preprint_v1.zip"
