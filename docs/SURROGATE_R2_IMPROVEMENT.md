# Surrogate R² improvement pipeline

**Goal:** Improve holdout `val_r2` and keep `ranking_wins: true` for the perturb-only pre-filter.

**Honest expectation:** With ~100–200 perturb labels, R² may stay negative; ranking can still win. Positive R² likely needs **300+** targeted labels (AL + merged search trials).

---

## Quick start (production — recommended)

```bash
cd ~/nanophotonics-inverse-design
source .venv/bin/activate

# Merge → AL 2–5 → surrogate_production → 5× sim-budget replicates + error bars
bash scripts/run_production_pipeline.sh

# Replicates only (after surrogate_production exists):
bash scripts/run_production_pipeline.sh --only-replicates
```

Legacy (variant comparison only):

```bash
python scripts/merge_search_labels_into_corpus.py
bash scripts/run_surrogate_improvement.sh
RUN_AL=1 bash scripts/run_surrogate_improvement.sh
```

---

## Variants (configs)

| Config | Architecture | Target | Output dir |
|--------|--------------|--------|------------|
| `configs/wedge_a.yaml` | `latent_mlp` | `split_ratio_upper` | `data/phase1/wedge_a/surrogate` |
| `configs/wedge_a_mask.yaml` | `mask_mlp` | `split_ratio_upper` | `data/phase1/wedge_a/surrogate_mask_perturb` |
| `configs/wedge_a_mask_abserr.yaml` | `mask_mlp` | `abs_split_error` | `data/phase1/wedge_a/surrogate_mask_abserr` |

Training uses `source_filter: perturb_plus_search` (batch perturb + merged MEEP search trials).

**Best historical R²:** `mask_mlp` + `perturb` only (~**−0.42** on 101 samples).  
**Do not** train on `all` (perlin+perturb) if the search stays in the σ-ball.

---

## What each step does

### Merge labels (`merge_search_labels_into_corpus.py`)

Adds rows from:

- `meep_search_local` / `meep_search_deep` / `meep_search_100` trials and top_candidates

into `data/phase0/sim_results_phase0_v1_all.csv` (backup → `.csv.bak`).

### Train (`train_wedge_a_surrogate.py`)

Reads `surrogate.target`, `hidden`, `max_iter`, `mask_pool` from YAML.

### `abs_split_error` target

Trains on `|split_ratio_upper − 0.5|`. Ranking and candidate generation use the same score (lower = better).

### Active learning (`run_wedge_a_al_batch.py`)

```bash
python scripts/run_wedge_a_al_batch.py --config configs/wedge_a_mask.yaml --from-round 2 --to-round 4
```

Each round: 1000 proposals → MEEP top 15 → merge corpus → retrain.

---

## Sim-budget B=50 / B=100 (separate track)

Does **not** improve R² unless you merge those MEEP rows into the corpus.

```bash
# After training mask_mlp ranker (recommended)
python scripts/train_wedge_a_surrogate.py --config configs/wedge_a_mask.yaml
bash scripts/run_sim_budget_b50_b100.sh
```

Or full study:

```bash
bash scripts/run_full_sim_budget.sh
```

---

## Metrics to watch

| Metric | File | Gate |
|--------|------|------|
| `val_r2` | `*/surrogate/metrics.json` | Internal; aim > 0 |
| `ranking_wins` | `ranking_eval_*.json` | **true** for pre-filter |
| Comparison | `data/phase1/surrogate_variant_comparison.csv` | Pick best R² with ranking_wins |

---

## Related

- [VALUE_PROPOSITION.md](VALUE_PROPOSITION.md) — pitch MEEP-native, not R²
- [WEDGE_A.md](WEDGE_A.md) — sim-budget wedge
