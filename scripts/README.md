# Scripts (Phase 0)

| Script | Sprint | Purpose |
|--------|--------|---------|
| `setup.sh` | 0 | Python 3.12.12 venv + install `drcgenerator` |
| `verify_setup.py` | 0 | Smoke test decode + heuristic DRC |
| `decode_batch.py` | 1 | Sample latents → masks → `data/phase0/manifest.csv` |
| `run_fdtd_batch.py` | 2 | MEEP labels → `data/phase0/sim_results.csv` (conda env `mp`) |
| `train_surrogate.py` | 3 | Fit latent → `split_ratio_upper` MLP on `sim_results.csv` |
| `latent_search.py` | 3 | Optuna BO over latent → surrogate, export top-k for MEEP |
| `evaluate_phase0.py` | 4 | Gate metrics JSON (search vs random, surrogate quality) |
| `active_learning_round.py` | 4+ | BO → MEEP top-k → retrain (closed loop) |

**Sprint 1 example:**

```bash
source ~/nanophotonics-inverse-design/.venv/bin/activate
uv pip install -r requirements-phase0.txt --python .venv/bin/python
python scripts/decode_batch.py --n-samples 500 --preview-png
```

Outputs: `data/phase0/manifest.csv`, `data/phase0/drc_report.json`, `data/phase0/masks/`, `data/phase0/latents/`.

**Sprint 3 example (after MEEP labels exist):**

```bash
source ~/nanophotonics-inverse-design/.venv/bin/activate
uv pip install -r requirements-phase0.txt --python .venv/bin/python
python scripts/train_surrogate.py
# partial batch (e.g. 83/500 ok): lower threshold for a smoke train
python scripts/train_surrogate.py --min-ok 50
```

Outputs: `data/phase0/surrogate/surrogate.joblib`, `metrics.json`, `training_rows.csv`, `train_summary.json`.

**Sprint 3 search (after surrogate trained):**

```bash
python scripts/latent_search.py --n-trials 200
# MEEP-verify top candidates (~20 by default)
bash scripts/run_meep.sh scripts/run_fdtd_batch.py \
  --manifest data/phase0/search/top_candidates.csv --no-skip-existing
```

Outputs: `data/phase0/search/search_trials.csv`, `top_candidates.csv`, `candidates/{latents,masks}/`.

**Sprint 4 gate + improvements:**

```bash
python scripts/evaluate_phase0.py
# Retrain mask surrogate on perturb-only (recommended)
python scripts/train_surrogate.py --architecture mask_mlp --sources perturb \
  --output-dir data/phase0/surrogate_mask_perturb

# Re-sim broken row
bash scripts/run_meep.sh scripts/run_fdtd_batch.py \
  --sample-ids perlin_00122 --force-resim

# Active learning round (BO → MEEP 5 → retrain)
python scripts/active_learning_round.py --round 1
```

See [docs/phase0_results.md](../docs/phase0_results.md).

**Engineering Sprint V1 (fundamental):**

```bash
# G1: does ref_published hit ~0.5 in-template?
bash scripts/run_meep.sh scripts/calibrate_meep.py --recipe-version phase0_v0 --verbose
bash scripts/run_meep.sh scripts/calibrate_meep.py --recipe-version phase0_v1 --try-flip-y --verbose

# Relabel under v1 (separate CSV)
bash scripts/run_meep.sh scripts/relabel_recipe.py --limit 101

# True inverse search (MEEP = objective)
bash scripts/run_meep.sh scripts/meep_search.py --n-trials 30
```

See [docs/ENGINEERING_SPRINT_V1.md](../docs/ENGINEERING_SPRINT_V1.md).

**Phase 0 closeout (best design `meep_bo_00010`):**

```bash
bash scripts/run_phase0_closeout.sh
# or: python scripts/finalize_phase0.py && python scripts/export_best_design.py
```

See [docs/phase0_results.md](../docs/phase0_results.md).

**Phase 1 week 2 (scale v1 labels + ranking + local σ refine):**

```bash
bash scripts/run_phase1_week2.sh
bash scripts/run_meep.sh scripts/meep_search_local.py --n-trials 50
python scripts/evaluate_surrogate_ranking.py --surrogate data/phase0/surrogate_phase1_v1 \
  --sim-results data/phase0/sim_results_phase0_v1.csv --output data/phase1/surrogate_ranking_eval_101.json
```

See [docs/phase1_results.md](../docs/phase1_results.md) and [docs/PHASE1_GETTING_STARTED.md](../docs/PHASE1_GETTING_STARTED.md).

**Phase 1 deep dev (multi-objective + surrogate-ranked AL + GDS):**

```bash
uv pip install -r requirements-phase1-deep.txt --python .venv/bin/python  # optional
bash scripts/run_phase1_deep.sh
```

See [docs/PHASE1_DEEP_DEV.md](../docs/PHASE1_DEEP_DEV.md).

**Phase 1 Track B (structural improvements — all pillars):**

```bash
bash scripts/run_track_b.sh          # pilot (~30–60 min MEEP)
bash scripts/run_track_b.sh --full     # larger searches
```

See [docs/PHASE1_TRACK_B.md](../docs/PHASE1_TRACK_B.md).

**Wedge A (sim-budget inverse design — primary):**

```bash
bash scripts/run_wedge_a.sh          # pilot
bash scripts/run_wedge_a.sh --full   # budgets 30/50/100
```

See [docs/WEDGE_A.md](../docs/WEDGE_A.md).

**Pilot outreach package:**

```bash
bash scripts/run_pilot.sh --config configs/pilot/benchmark_50_50.yaml --report-only
bash scripts/run_pilot.sh --config configs/pilot/benchmark_50_50.yaml --full-meep --skip-train
```

See [docs/PILOT_README.md](../docs/PILOT_README.md).

**Design novelty vs ref_published:**

```bash
python scripts/characterize_design_novelty.py
# → data/phase1/novelty/novelty_summary.md, hamming_cdf.png
```

**Full sim-budget (required before efficiency outreach):**

```bash
bash scripts/run_full_sim_budget.sh   # MEEP budgets 30, 50, 100
```

See [docs/VALUE_PROPOSITION.md](../docs/VALUE_PROPOSITION.md).

**Surrogate R² + sim-budget B=50/100:**

```bash
bash scripts/run_surrogate_improvement.sh      # merge + train + compare
bash scripts/run_sim_budget_b50_b100.sh        # MEEP (needs conda mp)
```

See [docs/SURROGATE_R2_IMPROVEMENT.md](../docs/SURROGATE_R2_IMPROVEMENT.md).

**Production (AL + frozen surrogate + 5 replicate sim-budget studies):**

```bash
bash scripts/run_production_pipeline.sh
# docs/SIM_BUDGET_REPLICATION.md
```
