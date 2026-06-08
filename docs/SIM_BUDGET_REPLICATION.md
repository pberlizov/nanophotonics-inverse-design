# Sim-budget replication protocol

**Purpose:** Five independent runs with error bars on `n_in_spec` and `best_abs_err` before external demo.

## Frozen across replicates

| Item | Value |
|------|--------|
| MEEP recipe | `phase0_v1`, resolution 25 |
| Surrogate weights (5× study) | `data/phase1/wedge_a/surrogate_production/surrogate.joblib` (frozen baseline) |
| Surrogate weights (current production) | `data/phase1/wedge_a/surrogate_improved/surrogate.joblib` |
| Budgets B | 30, 50, 100 |
| Policies | See `configs/wedge_a_production.yaml` |

## Varies per replicate `r ∈ {1..5}`

| Item | Rule |
|------|------|
| Seed | `base_seed + r * 1000` (default base 2026) |
| Candidate pool | Regenerated per replicate under `sim_budget/replicates/run_XX/` |
| σ / random paths | New Optuna + RNG draws |

## Does not vary

- Surrogate **not** retrained between replicates (stochasticity = search only).
- Corpus frozen after AL phase completes.

## Commands

```bash
# Full pipeline (AL → train → 5 replicates)
bash scripts/run_production_pipeline.sh

# Replicates only (uses surrogate.output_dir from wedge_a_production.yaml)
bash scripts/run_sim_budget_replicates.sh

# Validate promoted ranker (one run, B=100 only)
bash scripts/run_validation_replicate.sh

# Aggregate + plots
python scripts/aggregate_sim_budget_replicates.py
```

## Outputs

| Path | Content |
|------|---------|
| `sim_budget/replicates/run_01/` … `run_05/` | Per-run MEEP CSVs + `run_report.json` |
| `sim_budget_replicates.json` | All runs |
| `sim_budget_replication_report.md` | Tables mean ± std |
| `sim_budget_replication_errorbars.png` | Slide figure |

## Results (2026-05-26) — complete

Full tables, interpretation, and outreach phrasing: **[SIM_BUDGET_REPLICATION_RESULTS.md](SIM_BUDGET_REPLICATION_RESULTS.md)**.

**B=100 (5 seeds, mean ± std):**

| Policy | n_in_spec | best_abs_err |
|--------|-----------|--------------|
| **surrogate_rank** | **15.0 ± 3.2** | **0.0042 ± 0.0031** |
| hierarchical_35 | 14.8 ± 3.5 | 0.0052 ± 0.0029 |
| random_perturb | 12.8 ± 1.8 | 0.0058 ± 0.0061 |
| sigma_meep | 10.6 ± 4.8 | 0.0074 ± 0.0070 |

## Claims enabled

- “At **B=100**, surrogate-ranked search yields **~15 ± 3** in-spec designs vs **~11 ± 5** for σ-only (`sigma_meep`) over 5 seeds.”
- “Best promoted split error **~0.004 ± 0.003** (mean best \|split−0.5\|) for `surrogate_rank` at B=100.”
- Do **not** claim surrogate beats `hierarchical_35` on count at B=100 without noting overlapping CIs (14.8 ± 3.5 vs 15.0 ± 3.2).
