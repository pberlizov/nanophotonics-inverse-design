# Phase 1 — Getting started

**Status:** Started 2026-05-25  
**Phase 0 gate:** **Go (in-template)** — see [phase0_results.md](phase0_results.md)  
**Champion design:** `meep_bo_00093` @ MEEP split **0.497** (`phase0_v1`)

Phase 0 proved **MEEP-native search** works. Phase 1 scales labels, tightens the simulation contract, and only re-enables surrogate-assisted search when the forward model is calibrated.

---

## Phase 0 handoff (frozen references)

| Artifact | Path |
|----------|------|
| Best mask / latent | `data/phase0/meep_search_100/candidates/` |
| v1 labels (perturb) | `data/phase0/sim_results_phase0_v1.csv` (101 rows) |
| Merged training set | `data/phase0/sim_results_phase0_final.csv` (111 rows) |
| MEEP recipe | `phase0_v1` in [configs/phase0.yaml](../configs/phase0.yaml) |
| Do **not** use for design | v0 `sim_results.csv` + latent-surrogate BO |

---

## Phase 1 goals (4–8 weeks)

| Track | Goal | Exit |
|-------|------|------|
| **1A — Labels** | Full perturb corpus @ `phase0_v1` | 500+ ok rows, frozen `recipe_version` |
| **1B — Search** | MEEP-native BO at scale | ≥3 designs in-spec; best split &lt; 0.02 from 0.5 |
| **1C — Surrogate** | Mask forward model (optional BO) | Holdout R² &gt; 0 on v1 corpus before BO-on-surrogate |
| **1D — Template** | Align ref / fab narrative | Document template vs paper FDTD gap |
| **1E — Fab prep** | Export + MPW discussion | GDS/layout from champion mask |

---

## Week 1 checklist (start here)

### Day 1 — Export + freeze champion

```bash
source .venv/bin/activate
python scripts/export_best_design.py --sample-id meep_bo_00093
# → data/phase0/exports/meep_bo_00093_mask.png
```

Commit tag suggestion: `phase0-gate-meep_bo_00093`.

### Day 1–2 — Full v1 relabel (~15 min MEEP)

```bash
bash scripts/run_meep.sh scripts/relabel_recipe.py --sources perturb
# → data/phase0/sim_results_phase0_v1.csv (refresh to ~500 rows)
```

Merge any new `meep_bo_*` from search:

```bash
python scripts/finalize_phase0.py --skip-train
python scripts/train_surrogate.py \
  --manifest data/phase0/manifest_phase0_final.csv \
  --sim-results data/phase0/sim_results_phase0_final.csv \
  --architecture mask_mlp --output-dir data/phase0/surrogate_phase1_v0
```

**Gate:** Only use surrogate BO if `val_r2 > 0` in `train_summary.json`.

### Day 2–3 — MEEP search at scale

```bash
bash scripts/run_meep.sh scripts/meep_search.py --n-trials 200 \
  --output-dir data/phase1/meep_search
```

Label top-20, resim best:

```bash
bash scripts/run_meep.sh scripts/run_fdtd_batch.py \
  --manifest data/phase1/meep_search/top_candidates.csv --force-resim
```

### Week 2 — Scale + measure (`run_phase1_week2.sh`)

```bash
bash scripts/run_phase1_week2.sh
```

Adds:

- `sim_results_phase0_v1_all.csv` — full 500 @ v1  
- `evaluate_surrogate_ranking.py` — surrogate top-k vs random on MEEP  
- `meep_search_local.py` — refine σ around champion  
- `data/phase1/surrogate_mask_v1_full/`

### Day 4+ — Active learning (when surrogate calibrates)

```bash
python scripts/active_learning_round.py --round 1 \
  --surrogate-in data/phase0/surrogate_phase1_v0/surrogate.joblib
```

Use **MEEP verify** on every promoted design until surrogate R² is stable.

---

## What we are not doing in Phase 1 week 1

- Latent MLP on mixed perlin/perturb v0 labels  
- Surrogate BO without MEEP confirmation  
- Requiring `ref_published` ≈ 0.5 in-template (wrong gate — see calibration doc)

---

## Config changes for Phase 1

Create `data/phase1/` and optionally `configs/phase1.yaml` when:

- Recipe bumps to `phase0_v2` (template/port tuning), or  
- Platform adds photolithography manifold.

Until then, keep `phase0_v1` and write new CSVs under `data/phase1/`.

---

## Success metrics (Phase 1 gate)

| Metric | Target |
|--------|--------|
| Best MEEP split | 0.48–0.52, confirmed by 2× resim |
| In-spec rate @ 200 trials | ≥10% (vs 12% @ 100 trials) |
| Surrogate R² (mask_mlp, v1 full corpus) | &gt; 0.3 |
| Fab | Layout exported for ≥1 champion |

---

## Links

- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) — architecture  
- [ENGINEERING_SPRINT_V1.md](ENGINEERING_SPRINT_V1.md) — why MEEP-native won  
- [ROADMAP.md](ROADMAP.md) — milestones  
- [MEEP_SETUP.md](MEEP_SETUP.md) — two-env workflow (`.venv` decode, `mp` sim)
