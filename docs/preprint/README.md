# Preprint draft (`docs/preprint/`)

ArXiv-style draft for the nanophotonics inverse-design wedge (frozen MEEP contract + DRC manifold search + ranking surrogate pre-filter). **Simulation-only preprint** — not a foundry or experimental paper.

## Compile

From this directory:

```bash
cd docs/preprint
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

Requirements: standard TeX Live (`article`, `hyperref`, `graphicx`, `booktabs`).

### Figures

Regenerate from repo data:

```bash
python scripts/characterize_design_novelty.py
python scripts/aggregate_sim_budget_replicates.py --config configs/wedge_a_production.yaml
python scripts/export_preprint_figures.py
```

Outputs in `docs/preprint/figures/`:

- `hamming_cdf.pdf` / `.png`
- `sim_budget_curve.pdf` / `.png` (also `sim_budget_replication_errorbars.*`)

## Claims we make

1. **Frozen MEEP contract (`phase0_v1`).** All primary optical numbers use 2D TE, 25 px/µm, `mask_flip_y=true`, 180×180 mask on 4 µm island / 6 µm cell.
2. **Functional gap vs published reference.** `ref_published` → \(R_{\mathrm{up}} \approx 0.61\); champions reach \(\approx 0.50\) in the same recipe.
3. **Nonlinear mask→split map.** σ-local champion `local_00022` is ~0.47% Hamming from ref; random σ perturbations ~3.4% mean Hamming but ~3% in-spec rate.
4. **Far-manifold search.** In-spec Perlin samples ~51% Hamming from ref (outside expert σ ball).
5. **MEEP promotes; surrogate pre-filters only.** Lead champion `cand_000261`: prod r25 = 0.4946 from rank-surrogate MEEP-gated shortlist (`round_rank`, seed 2033, 500 proposals, top-15 MEEP).
6. **Mesh audit.** Six champions pass `phase0_v1_sdf_geom` dual gate (0.500/0.500 @ r25 & r50, zero mesh gap on panel).
7. **Sim-budget replication (5 seeds).** At B=100: `surrogate_rank` 15.0±3.2 in-spec (tied with `hierarchical_35` 14.8±3.5), best mean |err| 0.0042±0.0031; ahead of `sigma_meep` and `hierarchical_65` on reported aggregates.
8. **Ranking > regression for ML gate.** Pairwise rank loss improves Spearman on |err|; holdout R² remains negative (~−0.5 to −0.6 for MSE baseline).

## Claims we do **not** make

- Foundry DRC sign-off or fabrication yield / wafer correlation
- Surrogate replaces MEEP or “signs off” on designs
- Universal superiority over all baselines at every budget (B=30 weak; B=50 count favors `hierarchical_35`)
- Positive val R² as a product metric
- End-to-end differentiable manifold optimization as in Danis et al. (cited as related work)
- Mesh independence for all candidates (extended panel includes failures)
- invrs-gym / Ceviche scores as primary results (Track C is orthogonal; FDFD ≠ MEEP)

## Track C (appendix result, 2026-06-07)

On `ceviche_lightweight_power_splitter`, latent refine beats published leaderboard **0.0090** → **0.0010** (`meep_bo_00128`). See `data/phase1/invrs_benchmark/leaderboard_comparison.md`.

## Source artifacts in repo

| Topic | Path |
|-------|------|
| Novelty / Hamming | `data/phase1/novelty/novelty_summary.md` |
| MEEP recipe | `docs/sim_recipe_phase0.md`, `configs/phase0.yaml` |
| Sim-budget stats | `docs/SIM_BUDGET_REPLICATION_RESULTS.md` |
| Promotion / mesh | `data/phase1/meep_research/promotion_validation.md` |
| Wedge architecture | `docs/WEDGE_A.md` |
| Rank round vs regression | `data/phase1/wedge_a/meep_gated_shortlist/round_rank/vs_regression_round5.md` |

## Authors / affiliation

Placeholder author block in `manuscript.tex` — replace before arXiv submission.

## Bibliography notes (2026-06-02)

`refs.bib` was corrected against primary sources: Hammond et al.\ (Opt.\ Express 2022, DOI `10.1364/OE.442074`), Danis et al.\ (arXiv:2602.03142), Schubert et al.\ for Ceviche challenges (ACS Photonics 2022), Park et al.\ CNN-IBO (Sci.\ Rep.\ 2025), Ma et al.\ MAPS (DATE 2025 / arXiv:2503.01046), Schubert for invrs-gym (arXiv:2410.24132).
