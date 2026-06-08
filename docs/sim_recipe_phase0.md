# Phase 0 simulation recipe — MEEP 2D TE

**Status:** Template v0 — consistent batch labels for surrogate training, not a foundry sign-off model.

**Solver:** MEEP 2D TE (`Ez` polarization), effective-index style slab (`dimensions=2`).

---

## Purpose

Map each decoded binary mask from `drcgenerator` to scalar figures of merit:

| Field | Meaning |
|-------|---------|
| `flux_in` | Power flux at input monitor (a.u.) |
| `flux_out_upper` | Upper output arm @ 1550 nm |
| `flux_out_lower` | Lower output arm @ 1550 nm |
| `split_ratio_upper` | `flux_out_upper / (flux_out_upper + flux_out_lower)` |
| `insertion_loss_db` | `-10 log10((P_out_u + P_out_l) / P_in)` (approximate) |

Target for inverse design: `split_ratio_upper ≈ 0.5` at **1550 nm**.

---

## Geometry template (fixed for all masks)

All masks are **180×180** on a **4 µm × 4 µm** design island centered in a **6 µm × 6 µm** cell.

```
        upper out arm  (x > +2 µm, y ∈ [+0.35, +0.85])
              ↑
  input ──► [  4×4 µm design region (mask)  ]
              ↓
        lower out arm  (x > +2 µm, y ∈ [-0.85, -0.35])
```

- **Input waveguide:** Si strip, `x < -2 µm`, `|y| < wg_width/2`, connects into design.
- **Design region:** `x,y ∈ [-2, +2] µm` — permittivity from mask (1 → Si, 0 → SiO₂).
- **Output arms:** Si strips on the right; flux monitors sit in the arms.

This template is a **1×2 splitter topology**. It may not match every arbitrary mask topology from the generative model, but it is **fixed across the dataset**, which is what the surrogate needs for Phase 0.

---

## Materials (1550 nm, simplified)

| Region | ε |
|--------|---|
| Si | 12.0 (~n≈3.46) |
| SiO₂ background | 2.25 (~n=1.5) |

---

## Source and monitors

- **Source:** Gaussian `Ez` line source at input center, frequency `fcen = 1/1.55` µm⁻¹.
- **PML:** 0.5 µm on all sides.
- **Resolution:** default **25 pixels/µm** (`configs/phase0.yaml` → `meep.resolution`).  
  - Pilot: `--resolution 15` or `meep.resolution: 15` for speed.  
  - Production labels: 20–30.

---

## Runtime

- `until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(...), 1e-4)`  
- Typical wall time: **~30 s–3 min / mask** at resolution 25 on a laptop (varies widely).

---

## Mesh-stable verification (`phase0_v1_sdf_geom`)

Promoted mesh-stable recipe (2026-06): **`phase0_v1_sdf_geom`** with `sdf_smooth_um=0.04`.

- Analytical Si port blocks + signed-distance smooth ε in the design region
- Passes dual gate on three champions: |r25−r50| ≤ 0.03 and |r25−prod| ≤ 0.03
- Use for **r50 confirmation** and resolution cross-checks
- **Do not** relabel historical `phase0_v1` corpus rows

Legacy fallback: `phase0_v1_refgrid100n` (mesh-stable but large prod offset).

## Known limitations (v0)

1. **2D TE** — no 3D vectorial effects, no 220 nm slab mode dispersion modeled explicitly.
2. **Fixed ports** — masks optimized in another FDTD setup in the paper may not align with this template; labels are **self-consistent**, not absolute truth vs experiment.
3. **Heuristic DRC ≠ sim-ready** — 500/501 heuristic pass does not imply good split ratio.
4. Recipe should be **frozen** for the first surrogate; change only with a new `recipe_version` column.

---

## Config keys

See `configs/phase0.yaml` section `meep:`.

Implementation: `src/nano_inv/meep_sim.py`, runner: `scripts/run_fdtd_batch.py`.
