# Phase 0 results — EBL 50/50 power splitter

**Last updated:** 2026-05-25  
**Primary recipe:** `phase0_v1` (MEEP 2D TE, flux-decay runtime, `mask_flip_y: true`)  
**Manifold:** `drcgenerator` EBeamModel  

---

## 1. Executive summary

| Sprint | Status | Outcome |
|--------|--------|---------|
| 0–1 Manifold + decode | Done | 501 masks, 500/501 heuristic DRC pass |
| 2 MEEP labels (v0) | Done | 500/500 ok (~15 min @ res 25) |
| 3 Surrogate + latent BO | Done | Pipeline demo; **surrogate not calibrated** (R² &lt; 0) |
| 4 Engineering Sprint V1 | Done | **MEEP-native search works** |
| Gate | **Go (in-template)** | Best design **meep_bo_00093** @ **0.497** (100-trial MEEP search) |

**Bottom line:** Inverse search toward 50/50 **succeeds when MEEP is the objective**. Surrogate-first BO on v0 labels does not. Phase 1 should scale **meep_search** + v1 label corpus, not latent-surrogate BO alone.

---

## 2. Best design (fab candidate for template validation)

| Field | Value |
|-------|-------|
| **sample_id** | **`meep_bo_00093`** (100-trial search) |
| **recipe_version** | `phase0_v1` |
| **split_ratio_upper (MEEP)** | **0.497** |
| **σ (perturbation)** | 0.023 |
| **In-spec** (0.45–0.55) | Yes |
| **Mask** | `data/phase0/meep_search_100/candidates/masks/meep_bo_00093_mask.npy` |
| **Latent** | `data/phase0/meep_search_100/candidates/latents/meep_bo_00093_latent.npy` |

**Earlier champion (30-trial):** `meep_bo_00010` @ 0.529 (resim confirmed).

**Other in-spec (100-trial top-5):** `meep_bo_00055` (0.504), `meep_bo_00054` (0.494), `meep_bo_00038` (0.508), `meep_bo_00041` (0.519).

```bash
python scripts/export_best_design.py --sample-id meep_bo_00093 \
  --mask data/phase0/meep_search_100/candidates/masks/meep_bo_00093_mask.npy
```

---

## 3. MEEP-native search

| Run | Trials | In-spec | Best ID | Best split | σ |
|-----|--------|---------|---------|------------|---|
| `meep_search/` | 30 | 4/30 | `meep_bo_00010` | 0.529 | 0.053 |
| **`meep_search_100/`** | 100 | **12/100** | **`meep_bo_00093`** | **0.497** | 0.023 |

Compare to **surrogate BO** (`search_00089`): pred 0.499 → MEEP **0.078**.

---

## 4. Datasets

| File | Rows | Notes |
|------|------|-------|
| `manifest.csv` | 501 | Original decode batch |
| `sim_results.csv` | 525+ | v0 + search + **meep_bo_*** |
| `sim_results_phase0_v1.csv` | 101 | Perturb cohort @ v1 recipe |
| `sim_results_phase0_final.csv` | 111 | v1 perturb + 10 meep_native |
| `manifest_phase0_final.csv` | 118 | perturb + meep_search top-10 |

**Calibration (G1):**

- **G1a pass:** empty/full masks → 0.50 (template symmetric).
- **G1b fail (expected):** `ref_published` → ~0.33 (normal) / ~0.61 (`flip_y`); mask tuned for another solver, not this template.

---

## 5. Surrogate experiments (do not use for design sign-off)

| Model | Corpus | Val MAE | Val R² |
|-------|--------|---------|--------|
| latent_mlp | v0 all (499) | 0.355 | −0.99 |
| mask_mlp | perturb v0 (101) | 0.356 | −2.06 |
| mask_mlp | v1 + AL (106) | 0.272 | −1.34 |
| mask_mlp | v1 only (101) | 0.285 | −0.42 |

Surrogate may still help **ranking** after retrain on `sim_results_phase0_final.csv`; verify on MEEP before trusting BO.

---

## 6. Go / no-go

| Criterion | Result |
|-----------|--------|
| End-to-end pipeline | **Pass** |
| ≥1 MEEP in-spec design | **Pass** (`meep_bo_00093` @ 0.497) |
| Resim reproducibility | **Pass** |
| MEEP-native search &gt; surrogate BO | **Pass** |
| Surrogate R² &gt; 0.3 | **Fail** (Phase 1) |
| ref_published ≈ 0.5 in-template | **Fail** (expected; different FDTD setup) |

**Verdict: Go to Phase 1** for in-template inverse design; document template≠paper FDTD for fab.

---

## 7. Closeout commands

```bash
# Full automation (~10 min MEEP search + train + export)
bash scripts/run_phase0_closeout.sh

# Or stepwise:
bash scripts/run_meep.sh scripts/meep_search.py --n-trials 100 --output-dir data/phase0/meep_search_100
source .venv/bin/activate
python scripts/finalize_phase0.py
python scripts/export_best_design.py --sample-id meep_bo_00010
python scripts/evaluate_phase0.py --output data/phase0/gate_metrics_final.json
```

---

## 8. Phase 1 priorities

1. Scale **meep_search** (100–200 trials) on `phase0_v1`.
2. Relabel full perturb corpus → `sim_results_phase0_v1.csv` (500 rows, ~15 min).
3. Retrain **mask_mlp** on merged corpus; only re-enable surrogate BO if R² &gt; 0.
4. Template/port alignment study if fab needs match to published 50/50 ref.
5. Optional MPW: export GDS from `meep_bo_00010` mask.

---

## 9. Artifacts

| Path | Description |
|------|-------------|
| `data/phase0/meep_search/` | 30-trial MEEP BO + top-10 |
| `data/phase0/phase0_closeout.json` | Merge + train summary |
| `data/phase0/surrogate_phase0_final/` | Final mask_mlp |
| `data/phase0/exports/` | Best-design PNG + JSON |
| `docs/ENGINEERING_SPRINT_V1.md` | Sprint plan |
