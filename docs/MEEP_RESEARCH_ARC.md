# MEEP research arc — sim contract & tool value

**Status:** active (2026-05-30)  
**Production contract (unchanged):** `phase0_v1` @ r25, `mask_flip_y: true`  
**Promotion gate:** PASSED — `phase0_v1_sdf_geom` on champion panel (2026-06)

---

## Why this matters

The surrogate/search stack is only as valuable as the **frozen forward model**. Until MEEP labels are mesh-stable and externally trend-checked, we cannot claim:

- resolution-independent verification  
- reliable high-res confirmation runs  
- cross-solver sign-off beyond trend checks  

This arc runs **in parallel** with outreach on the narrow r25 contract.

---

## What we know (baseline)

| Finding | Evidence |
|---------|----------|
| r50 ≠ r25 on production | 0.09–0.14 gap on 3 champions |
| Not runtime / flux-decay | `v1_fixed` identical to production |
| Not `eps_averaging` alone | `noavg` unchanged at r50 |
| HDF5 `epsilon_input_file` | r25 ≈ prod; **r50 worse** (0.25 gap) |
| `matgrid` (design-only) | best **mean** mesh gap (~0.07); r25 offset ~0.05 |
| `refgrid100n` | mesh-stable on 1/3 masks; large r25 offset |

See `docs/RECIPE_SENSITIVITY.md`, `data/phase1/recipe_sensitivity/mesh_stable_b_report.md`.

---

## Research phases

### Phase 1 — Infrastructure (done / in progress)

- [x] Recipe variants in `src/nano_inv/meep_sim.py` (matgrid, refgrid, epsfile, fullcell_matgrid)
- [x] Champion sensitivity scripts (`study_mesh_stable_b.py`, `study_meep_research.py`)
- [x] Config-driven experiment matrix (`configs/meep_research.yaml`)
- [ ] CI-style regression: 3 champions × production r25 on every MEEP-touching PR

### Phase 2 — Geometry & discretization (current focus)

**Hypothesis:** mesh gaps come from **inconsistent port/design junction** when ε is built differently at each resolution.

| Experiment | Recipe | Goal |
|------------|--------|------|
| Full-cell raster | `phase0_v1_fullcell_matgrid` | Single fixed 100 px/µm grid for **ports + mask** |
| Matgrid + avg | `phase0_v1_fullcell_matgrid_avg` | Subpixel edge smoothing on full-cell grid |

**Early result (`local_00022`):** fullcell r25 **0.471** (|Δprod|≈0.028, better than design-only matgrid) but r50 gap **0.21** (worse). Design-only `matgrid` still leads on mesh stability (gap ~0.10).

| Matgrid calibration | sweep `arm_y`, `wg_width`, port overlap | Minimize \|r25 − prod\| at fixed mesh gap |
| Per-champion sweeps | `configs/matgrid_calibration_{id}.yaml` | `local_00022`, `meep_bo_00093`, rebuild `meep_bo_00128` |
| Aggregate panel | `scripts/aggregate_matgrid_calibration.py` | Triple-pass gate across 3 champions |
| epsfile @ r25 freeze | freeze HDF5 at 25 px/µm (not 100) | Test whether file res matches production |

### Phase 3 — Template & broadband

- Calibrate `phase0_v2` geometry vs `ref_published` (arm spacing, wg width)
- C-band sweep 1500 / 1550 / 1600 nm on promoted recipe
- Re-run Tidy3D on promoted recipe (3 champions, frugal mode)

### Phase 4 — Production integration

- Add `recipe_version` column for **new** labels only (no silent corpus rewrite)
- Promotion script updates `configs/phase0.yaml` + `SIM_CONTRACT.md`
- Optional: client-specific calibration YAML (paid track)

---

## How to run

```bash
# Full research panel (reads configs/meep_research.yaml)
bash scripts/run_meep.sh scripts/study_meep_research.py

# One champion, fast iteration
bash scripts/run_meep.sh scripts/study_meep_research.py --sample-id local_00022

# Legacy sensitivity sweeps
bash scripts/run_meep.sh scripts/study_mesh_stable_b.py
```

Outputs: `data/phase1/meep_research/research_report.md`

---

## Success metrics (tool value)

| Metric | Target | Current |
|--------|--------|---------|
| Champion mesh gap (max) | ≤ 0.03 | **0.000 (`sdf_geom`)** |
| Champion \|r25 − prod\| (max) | ≤ 0.03 | 0.00 @ prod |
| Tidy3D vs MEEP r25 (max Δ) | ≤ 0.05 | ~0.05 (2/3) |
| Regression panel pass | 3/3 champions | prod only @ r25 |

Promoted via `scripts/promote_meep_recipe.py` → `phase0_v1_sdf_geom`.

---

## Outreach vs research

| Audience | Claim |
|----------|-------|
| **Today (email)** | MEEP-qualified @ `phase0_v1` r25; Tidy3D trend on champions |
| **After promotion** | **`phase0_v1_sdf_geom`** mesh-stable on champions; r50 confirmation available |
| **Never without evidence** | “Any resolution”, “any solver agrees” |
