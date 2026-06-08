# Engineering Sprint V1 — Trustworthy labels, then search

**Trigger:** Phase 0 pipeline works, but `ref_published` → MEEP split **~0.33** (not ~0.5). Surrogate R² &lt; 0 and BO picks are uncorrelated with MEEP. Incremental AL cannot fix a broken **simulation contract**.

**Goal:** Frozen **`phase0_v1`** with sane template wiring, then **MEEP-native search** (no surrogate). Surrogate only after v1 labels look structured.

**Compute (your machine, res 25):** ~**1.8–2 s/mask** → 500 masks ≈ **15 min**; 30 `meep_search` trials ≈ **1 min**; 101 relabel ≈ **3 min**. Not hours unless you scale to thousands or much higher resolution.

---

## Success criteria

| Gate | Test | Pass |
|------|------|------|
| **G1a Template** | `calibrate_meep.py` | empty + full masks split **≈ 0.50** (ports OK) |
| **G1b Ref (info)** | same script | `ref_published` ∈ 0.40–0.60 — **often fails**; mask is from another FDTD setup (~0.33 normal, ~0.61 flip_y) |
| **G2 MEEP search** | `meep_search.py` 30–50 trials | ≥1 design MEEP split ∈ 0.45–0.55 |
| **G3 Surrogate (optional)** | train on `sim_results_phase0_v1.csv` | R² &gt; 0 before BO-on-surrogate |

**Do not block on G1b.** If G1a passes, proceed with relabel + `meep_search`.

---

## Sprint phases

### Phase A — MEEP recipe v1 (morning)

1. Run `scripts/calibrate_meep.py` on **v0** (baseline numbers).
2. Implement **`phase0_v1`** in `meep_sim.py`:
   - Flux-based run termination (2D-safe monitor point)
   - Longer minimum runtime
   - Optional `mask_flip_y` for grid alignment check
   - Record `recipe_version` on every row
3. Re-run calibration; iterate geometry until G1 or document blocker.

**Artifacts:** `configs/phase0.yaml` (`meep.recipe_version: phase0_v1`), `docs/sim_recipe_phase0.md` update.

### Phase B — Relabel cohort (midday)

```bash
bash scripts/run_meep.sh scripts/relabel_recipe.py --recipe-version phase0_v1 \
  --sample-ids ref_published --limit 50
# then expand to full perturb cohort when G1 passes
```

Writes `data/phase0/sim_results_v1.csv` (does not overwrite v0 corpus).

### Phase C — MEEP-native inverse search (afternoon)

```bash
bash scripts/run_meep.sh scripts/meep_search.py --n-trials 30 --mode perturb
```

Objective = **actual MEEP** `|split_ratio_upper − 0.5|`. No surrogate.

### Phase D — Surrogate (only if G1–G3 pass)

Train `mask_cnn` on v1 labels; compare to MEEP search baseline.

---

## What we stop doing

- BO on surrogate with v0 labels
- “Round N” active learning without `recipe_version` bump
- Treating `search_00089`-style surrogate picks as designs

---

## Decision tree after sprint

```
G1 pass?
  yes → relabel 200–500 masks (v1) → meep_search → optional CNN
  no  → template redesign spike (3D MEEP, port refactor, or import reference sim)
```

---

## Commands cheat sheet

```bash
# A: calibration
bash scripts/run_meep.sh scripts/calibrate_meep.py --verbose
bash scripts/run_meep.sh scripts/calibrate_meep.py --recipe-version phase0_v1 --verbose

# B: relabel
bash scripts/run_meep.sh scripts/relabel_recipe.py --limit 100

# C: true inverse search
bash scripts/run_meep.sh scripts/meep_search.py --n-trials 30

# Evaluate
python scripts/evaluate_phase0.py --sim-results data/phase0/sim_results_v1.csv
```
