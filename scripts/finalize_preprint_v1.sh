#!/usr/bin/env bash
# CPU-only v1 preprint finalize: manifest, figures, PDF, readiness.
# Safe to run while MEEP terminals are busy.
#
# Usage:
#   bash scripts/finalize_preprint_v1.sh
#   SKIP_PDF=1 bash scripts/finalize_preprint_v1.sh
#
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-.venv/bin/python}"
FIG="$REPO/docs/preprint/figures"
REL="$REPO/data/phase1/release"

step() { echo ""; echo "==> $*"; }

step "Repro manifest"
$PY scripts/build_repro_manifest.py

step "Plot-only figures from existing JSON (no MEEP)"
$PY scripts/champion_mesh_convergence.py --plot-only 2>/dev/null || true
$PY scripts/champion_fab_stress.py --plot-only 2>/dev/null || true

step "Copy release MEEP figures → docs/preprint/figures/"
mkdir -p "$FIG"
for base in champion_mesh_convergence champion_fab_stress; do
  for ext in pdf png; do
    src="$REL/${base}.${ext}"
    if [[ -f "$src" ]]; then
      cp -f "$src" "$FIG/"
      echo "    cp ${base}.${ext}"
    fi
  done
done

step "Export preprint figures (hamming, sim-budget, pipeline)"
$PY scripts/export_pipeline_figure.py
$PY scripts/export_preprint_figures.py

step "Broadband contribution figure"
$PY scripts/export_broadband_contribution_figure.py

step "Broadband hunt report (CPU merge)"
$PY scripts/broadband_hunt.py --stage report 2>/dev/null || true

if [[ -f "$REPO/data/phase1/wedge_a/sim_budget_replication_stats.csv" ]]; then
  step "Sim-budget stats already aggregated"
else
  step "Aggregate sim-budget (if any replicates exist)"
  $PY scripts/aggregate_sim_budget_replicates.py \
    --config configs/wedge_a_release_replication.yaml 2>/dev/null \
    || $PY scripts/aggregate_sim_budget_replicates.py \
    --config configs/wedge_a_production.yaml 2>/dev/null \
    || echo "    (no replicate stats yet)"
fi

if [[ "${SKIP_PDF:-0}" != "1" ]]; then
  step "Build manuscript.pdf"
  (cd docs/preprint && pdflatex -interaction=nonstopmode manuscript.tex >/dev/null 2>&1 && \
   pdflatex -interaction=nonstopmode manuscript.tex >/dev/null 2>&1) || {
    echo "    pdflatex had warnings — check docs/preprint/manuscript.log"
  }
  ls -la docs/preprint/manuscript.pdf
fi

step "Loss-aware report (if trials exist)"
if [[ -f "$REPO/data/phase1/loss_aware_hunt/loss_aware_trials.csv" ]]; then
  $PY scripts/loss_aware_report.py
fi
if [[ -f "$REPO/data/phase1/phase2_il_hunt/phase2_il_trials.csv" ]]; then
  $PY scripts/phase2_il_report.py
fi
if [[ -f "$REPO/data/phase1/loss_aware_hunt/loss_aware_trials.csv" ]]; then
  $PY scripts/export_loss_aware_figure.py 2>/dev/null || true
fi

step "Readiness report"
$PY scripts/check_preprint_v1_readiness.py || true
$PY scripts/check_publication_readiness.py 2>/dev/null || true

step "Zenodo bundle (PDF + figures + release MDs)"
bash scripts/build_zenodo_bundle.sh

echo ""
echo "v1 finalize done. See docs/ZENODO_RELEASE.md and docs/preprint/V1_BACKLOG.md."
