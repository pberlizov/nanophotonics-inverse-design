# Preprint release checklist (reviewer feedback)

> **Primary v1 driver:** [`V1_BACKLOG.md`](V1_BACKLOG.md) — status, commands, arXiv checklist.  
> **Automate:** `bash scripts/finalize_preprint_v1.sh` · `python scripts/check_preprint_v1_readiness.py`

Maps nine reviewer requests to repo work, paper edits, and rough MEEP cost.
**v1 can post as preliminary** when Tier A CPU artifacts are done and n=20 replication is labeled in progress.

## Tier A — Required before release

| # | Request | Status | Repo action | Paper change |
|---|---------|--------|-------------|--------------|
| 1 | **n=20 budget replicates + B=200** | 🔄 | `N_REPLICATES=20 bash scripts/run_release_replication.sh` · frozen `surrogate_improved` · `configs/wedge_a_release_replication.yaml` | Replace Table budget + Fig budget; add paired-win line |
| 2 | **Transmission / IL gates** | ✅ | `champion_fom_table.py` + **`flux_il_audit.py`** (monitor sweep) | Table + audit: IL diagnostic, not gate |
| 3 | **Wavelength sweep + broadband hunt** | 🔄 | `champion_broadband_sweep.py` ✅ · **`run_broadband_hunt.sh`** refine running | `BROADBAND_CONTRIBUTION.md` + figure |
| 4 | **Mesh audit extension** | ✅ | `champion_mesh_convergence.py` + `--plot-only` | Fig `champion_mesh_convergence.pdf` |
| 9 | **Repro manifest + regen** | ✅ | `build_repro_manifest.py` · `finalize_preprint_v1.sh` | Data availability paragraph |

## Tier B — Strongly recommended

| # | Request | Status | Repo action |
|---|---------|--------|-------------|
| 5 | **Fab / process variation** | ✅ | `champion_fab_stress.py` → `champion_fab_stress.md` | 
| 6 | **Surrogate validation regimes** | ✅ | `surrogate_validation_regimes.py` + `r2_deep_work.py` |
| 7 | **Ablations** | ✅ | `ablation_proposal_pool.py` |
| 8 | **Novelty rigor** | ✅ | `characterize_design_novelty.py --extended` |

## Tier C — Nice to have

- Input reflection monitor in `meep_sim.py` (not implemented; flux_in exists, backward flux TBD)
- B=250 saturation curve
- 3D MEEP cross-check (out of Phase 0 scope)

---

## 1. Budget study (n=20, B=200)

**Current:** 5–6 seeds, B∈{30,50,100}; surrogate_rank 15.0±3.2 vs hierarchical_35 14.8±3.5 — overlapping CIs.

**Target outputs:**
- `data/phase1/wedge_a/sim_budget_replication_stats.csv` (all budgets)
- `data/phase1/wedge_a/sim_budget_paired_wins.md` (e.g. surrogate_rank beats sigma_meep in X/20 at B=100)
- 95% CI columns in aggregate report

**Commands:**
```bash
cd ~/nanophotonics-inverse-design
source .venv/bin/activate
# ~ (20 × 4 budgets × 6 policies × ~100) MEEP calls if run serially per policy — plan multi-day run
N_REPLICATES=20 CONFIG=configs/wedge_a_release_replication.yaml \
  bash scripts/run_release_replication.sh
python scripts/aggregate_sim_budget_replicates.py --config configs/wedge_a_release_replication.yaml
```

**Frozen:** corpus CSV + `surrogate_improved/surrogate.joblib` (do not retrain between replicates).

**Paper text:** Report paired seed wins, not only mean±std. Soften “ahead of sigma_meep” unless paired test holds at p<0.05 or ≥15/20 wins.

---

## 2. Transmission / insertion-loss gates

**Current:** `flux_in`, `flux_out_*`, `insertion_loss_db` logged; many champion rows missing flux in corpus.

**Release gate proposal:**
| Criterion | Suggested threshold |
|-----------|---------------------|
| \|R_up − 0.5\| | ≤ 0.05 (unchanged) |
| T_total = (P_up+P_low)/P_in | ≥ 0.80 (tune after table) |
| IL | ≤ 1.5 dB (or document exclusion) |
| Reflection | TBD when monitor added |

**Script:** `scripts/champion_fom_table.py` → `data/phase1/release/champion_fom_table.md`

---

## 3. Wavelength sweep

**Existing:** `simulate_mask_broadband()` in `src/nano_inv/meep_sim.py`.

**Default grid:** 1530–1570 nm step 5 nm (verify); search uses step 10 nm.

**Outputs:**
- Baseline (narrowband champions): `champion_broadband.json` + `.md` + `.png`
- **Contribution track:** `bash scripts/run_broadband_hunt.sh` → `broadband_hunt.md`, `broadband_verify.json`, `docs/preprint/figures/broadband_contribution.png`

**Gate:** worst |R_up−0.5| ≤ 0.05 over band. Current champions: **0/5 pass** — hunt targets new broadband-flat designs.

---

## 4. Mesh audit

**Current:** sdf_geom r25/r50, six champions at exactly 0.500 — readers will ask if SDF smoothing hides geometry.

**Add:**
- Resolutions 25, 35, 50, 75 px/µm
- Pixel vs sdf_geom curves for 261, 00022, 00093, rnd_100_00016_rep02 (failure)
- Plot R_up vs resolution, not pass/fail only

**Script:** `scripts/champion_mesh_convergence.py`

---

## 5. Fabrication stress

Morph masks: binary dilation/erosion at 10/20/30 nm (convert nm → pixels at 22.22 nm/px on 180×180 @ 4 µm).

**Script:** `scripts/champion_fab_stress.py` → `data/phase1/release/champion_fab_stress.md` + `.json`

**MEEP required.** Dry-run: `python scripts/champion_fab_stress.py --dry-run`

Optional: waveguide width ±10/20 nm via recipe override (not implemented).

---

## 6. Surrogate validation (three regimes)

| Regime | Script | Purpose |
|--------|--------|---------|
| Random 80/20 | `r2_deep_work.py` baseline_random | Shows negative R² |
| Group by σ / latent family | `r2_deep_work.py` abs_err_group_sig | R² ≈ 0.45 diagnostic |
| Chronological | `surrogate_validation_regimes.py` | Train pre-AL, test post-AL rows |

**Output:** `data/phase1/release/surrogate_validation.md` + `.json` (CPU-only training/eval)

Report: Spearman, top-20/50 in-spec, enrichment vs random.

---

## 7. Ablations

Minimum matrix (same 500-proposal pool per seed):
- random top-B
- MSE surrogate top-B
- rank surrogate top-B
- oracle (sort by true MEEP after the fact)

Optional: no Perlin, σ-only, M∈{100,250,500,1000}.

**Script:** `scripts/ablation_proposal_pool.py` → `data/phase1/release/ablation_proposal_pool.md` + `.json`

**CPU-only.** Flags: `--fast` (500-row pool), `--train-mse`, `--pool-csv PATH`.

---

## 8. Novelty

**Script:** `python scripts/characterize_design_novelty.py --extended`

**Outputs:**
- `data/phase1/release/novelty_extended.md` + `.json`
- `data/phase1/release/novelty_panels/*_xor_panel.png` (champion vs ref_published XOR)
- Nearest corpus neighbor Hamming (latent L2 NN in sim corpus)

Base report (no `--extended`): `data/phase1/novelty/novelty_summary.md`

---

## 9. Reproducibility

`scripts/build_repro_manifest.py` writes:
- git commit, MEEP version, Python + key package versions
- SHA256 of corpus CSV and champion masks
- Frozen YAML paths

`scripts/regenerate_release_artifacts.sh` one-shot: ref_published, 100 σ perturb, champions, figures.

---

## Suggested execution order

**One-shot driver (skips MEEP by default):** `bash scripts/run_release_backlog.sh`  
**Include MEEP steps:** `RUN_MEEP=1 bash scripts/run_release_backlog.sh`

1. Repro manifest + champion FOM table (hours; FOM needs MEEP)
2. Broadband + mesh convergence on champions (hours–1 day; MEEP)
3. Launch n=20 budget replication (days, background; MEEP)
4. Fab stress + ablations (fab stress MEEP; ablations CPU)
5. Surrogate validation table + novelty panels (CPU)
6. Update `manuscript.tex` + re-export figures

## Honest scope note for arXiv

If n=20 budget is not finished, post as **preliminary** and state “five-seed pilot; 20-seed replication in progress.” Do not claim decisive policy ordering until Tier A #1 completes.
