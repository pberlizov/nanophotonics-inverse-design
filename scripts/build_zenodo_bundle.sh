#!/usr/bin/env bash
# Build Zenodo citation bundle: PDF, figures, release summaries, CITATION, LICENSE, manifest.
#
# Usage:
#   bash scripts/build_zenodo_bundle.sh
#
# Output:
#   data/phase1/release/zenodo_bundle/          (staging)
#   data/phase1/release/nanophotonics_preprint_v1.zip
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

BUNDLE="$REPO/data/phase1/release/zenodo_bundle"
ZIP="$REPO/data/phase1/release/nanophotonics_preprint_v1.zip"
FIG_SRC="$REPO/docs/preprint/figures"
PDF_SRC="$REPO/docs/preprint/manuscript.pdf"
RELEASE="$REPO/data/phase1/release"

RELEASE_MDS=(
  champion_fom_table.md
  flux_il_audit.md
  broadband_hunt.md
  loss_aware_hunt.md
  phase2_il_hunt.md
  champion_mesh_convergence.md
  champion_fab_stress.md
)

step() { echo ""; echo "==> $*"; }

step "Clean staging dir"
rm -rf "$BUNDLE"
mkdir -p "$BUNDLE/figures" "$BUNDLE/release"

step "Copy manuscript.pdf"
if [[ ! -f "$PDF_SRC" ]]; then
  echo "ERROR: missing $PDF_SRC — run bash scripts/finalize_preprint_v1.sh first" >&2
  exit 1
fi
cp -f "$PDF_SRC" "$BUNDLE/manuscript.pdf"

step "Copy preprint figures (PDF + PNG)"
if [[ -d "$FIG_SRC" ]]; then
  cp -f "$FIG_SRC"/*.{pdf,png} "$BUNDLE/figures/" 2>/dev/null || true
fi

step "Copy release summaries"
for md in "${RELEASE_MDS[@]}"; do
  src="$RELEASE/$md"
  if [[ -f "$src" ]]; then
    cp -f "$src" "$BUNDLE/release/"
    echo "    $md"
  else
    echo "    WARN: missing $md"
  fi
done

step "Copy CITATION.cff, LICENSE, repro_manifest.json"
cp -f "$REPO/CITATION.cff" "$BUNDLE/"
cp -f "$REPO/LICENSE" "$BUNDLE/"
if [[ -f "$RELEASE/repro_manifest.json" ]]; then
  cp -f "$RELEASE/repro_manifest.json" "$BUNDLE/"
else
  echo "ERROR: missing repro_manifest.json — run finalize_preprint_v1.sh" >&2
  exit 1
fi

step "Write bundle README.txt"
cat > "$BUNDLE/README.txt" <<'EOF'
MEEP-Gated Search on DRC Manifolds for Silicon 1×2 Power Splitters
Version: v1.0-preprint (simulation-only research release)

Contents:
  manuscript.pdf       — preprint PDF
  figures/             — paper figures (PDF + PNG)
  release/             — markdown audit tables and hunt summaries
  CITATION.cff         — citation metadata
  LICENSE              — MIT
  repro_manifest.json  — frozen recipe hashes and corpus SHA256

Source code and full MEEP corpora:
  https://github.com/pberlizov/nanophotonics-inverse-design

Claim contract: MEEP-gated surrogate search under frozen phase0_v1;
documented negative broadband/morph/IL results. Not deployment-ready
devices or foundry-calibrated IL.

See docs/OPEN_SOURCE_RELEASE.md in the GitHub repo.
EOF

step "Create zip"
rm -f "$ZIP"
(
  cd "$BUNDLE"
  zip -r -q "$ZIP" .
)
ls -la "$ZIP"
echo ""
echo "Zenodo bundle ready: $ZIP"
echo "Upload checklist: docs/ZENODO_RELEASE.md"
