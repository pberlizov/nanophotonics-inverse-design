# Recipe sensitivity (MEEP vs MEEP)

## Problem

On the same mask, **`phase0_v1` r25 vs r50** can differ by **~0.10–0.14** in upper-arm split. That is separate from the **Tidy3D** cross-solver check (~0.05 on 2/3 champions).

Until the mesh gap is small, do **not** claim mesh-independence or “any resolution agrees.”

## Root cause

Production uses `epsilon_func` with nearest-pixel mask indexing. MEEP **re-samples that function on a finer Yee grid at r50**, so the effective ε field changes even when the mask array is identical.

Runtime / flux-decay length is **not** the main driver (`phase0_v1_tcap` and `noavg` did not fix r50 on `local_00022`).

## Mesh-stable candidate (validated on `local_00022`)

| Recipe | r25 | r50 | r25↔r50 gap | vs production r25 |
|--------|-----|-----|-------------|-------------------|
| `phase0_v1` | 0.499 | 0.644 | **0.144** | — |
| `phase0_v1_matgrid` | 0.552 | 0.656 | 0.104 | 0.053 |
| **`phase0_v1_refgrid100n`** | 0.629 | 0.606 | **0.022** | 0.129 |

`phase0_v1_refgrid100n`: frozen ε on a **100 px/µm** grid (mask scale), nearest mask raster, bilinear ε lookup, `eps_averaging: true`.

**Tradeoff:** Excellent **r25↔r50** agreement; absolute split **≠** legacy `phase0_v1` corpus (~0.50). Use as a **verification** recipe, not a drop-in relabel of historical CSV rows.

## Promoted mesh-stable recipe (`phase0_v1_sdf_geom`)

- Analytical Si port blocks + SDF smooth ε in design region (`sdf_smooth_um=0.04`)
- **Triple-pass on champions:** 0.500 @ r25 & r50, gap 0.000, |r25−prod| ≤ 0.009
- Extended spot-check: `perlin_00018`, `meep_bo_00055` pass; corpus replicate masks may fail |r25−prod| (expected — different prod baseline)
- Configured as `mesh_stable_recipe_version` in `configs/phase0.yaml`
- **Production corpus labels unchanged** (`phase0_v1` @ r25)

Evidence: `data/phase1/meep_research/d0_geometry_report.md`, `promotion_validation.md`

## Option B sprint (2026-05-30)

| Recipe | local_00022 r25 | r50 | r25↔r50 | |r25−prod| |
|--------|-----------------|-----|---------|------------|
| production | 0.499 | 0.644 | 0.144 | — |
| `phase0_v1_fixed` (t=2520, no decay) | 0.499 | 0.644 | 0.144 | 0.000 |
| `phase0_v1_epsfile` (HDF5 @100) | 0.518 | 0.770 | 0.253 | 0.018 |
| `phase0_v1_epsfile_fixed` | 0.518 | 0.770 | 0.253 | 0.018 |

**B1 (`epsilon_input_file`):** r25 within ~0.02 of production, but **r50 still diverges** (~0.25 gap) — not a fix.

**B2 (fixed Meep time on production ε_func):** **no change** vs flux-decay — rules out runtime/convergence as the main cause.

Run full 3-champion sweep:

```bash
bash scripts/run_meep.sh scripts/study_mesh_stable_b.py
```

Output: `data/phase1/recipe_sensitivity/mesh_stable_b_report.md`

## Dual-contract policy (recommended)

| Role | Recipe | Resolution |
|------|--------|------------|
| **Labels / outreach / corpus** | `phase0_v1` | **25** only |
| **Mesh-stability audit** | **`phase0_v1_sdf_geom`** | 25 and 50 (dual gate ≤ 0.03) |
| Legacy fallback | `phase0_v1_refgrid100n` | mesh-stable but large prod offset |

Configured in `configs/phase0.yaml` as `recipe_version` and `mesh_stable_recipe_version`.

## Scripts

```bash
# Per-champion matgrid geometry (current focus)
bash scripts/run_matgrid_backlog.sh local_00022
bash scripts/run_matgrid_backlog.sh meep_bo_00093
bash scripts/run_matgrid_backlog.sh all   # sequential + aggregate

# Quick tune on one mask
bash scripts/run_meep.sh scripts/tune_matgrid_recipe.py

# All champions, recipe families
bash scripts/run_meep.sh scripts/study_recipe_sensitivity.py
```

Outputs: `data/phase1/meep_research/matgrid_calibration_{id}.md`, combined `matgrid_calibration_combined.md`

## What to claim

| OK | Not OK |
|----|--------|
| Split @ **`phase0_v1` r25 flip_y** (sim contract) | “Any resolution / any solver” |
| Mesh audit @ **`phase0_v1_refgrid100n`** r25≈r50 | Rewriting corpus splits under refgrid100n |
| Tidy3D trend check vs MEEP r25 | Full third-party sign-off on all masks |

## Promote a single recipe (only if both criteria pass)

Requires on all three champions: **|split_r25 − split_r50| ≤ 0.03** and **|split_r25 − v1_r25| ≤ 0.03**.

No candidate met both on `local_00022` as of 2026-05-30; keep dual contract until a unified recipe is found.

**Longer arc:** see [MEEP_RESEARCH_ARC.md](MEEP_RESEARCH_ARC.md).
