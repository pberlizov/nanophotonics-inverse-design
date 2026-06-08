# Phase 1 Track B — Structural improvements

**Goal:** Improve what we sell (the inverse-design loop) before fab — larger search space, cleaner surrogates, honest sim contract, measurable baselines.

---

## Tracks (all started)

| ID | What | Script | Output |
|----|------|--------|--------|
| **B1** | Latent MEEP search (not σ-only) | `latent_meep_search.py` | `data/phase1/track_b/latent_meep_search/` |
| **B2** | Split surrogate training | `train_track_b_surrogates.py` | `data/phase1/track_b/surrogates/*` |
| **B3** | Template + mask co-search | `meep_template_search.py` | `data/phase1/track_b/template_search/` |
| **B4** | Multi-objective / Pareto | `summarize_pareto_trials.py` | `pareto_*.json` |
| **B5** | Acquisition presearch | `surrogate_acquisition_search.py` | `data/phase1/track_b/acquisition/` |
| **B6** | Second target (70/30) | `configs/phase2_splitter_70_30.yaml` | `data/phase2/...` |
| **B7** | Baselines | `run_baselines.py` | `data/phase1/track_b/baselines/` |

**Orchestrator:** `bash scripts/run_track_b.sh` (pilot) or `bash scripts/run_track_b.sh --full`

---

## B1 — Latent modes

```bash
# Residual on first K latent dims (default)
bash scripts/run_meep.sh scripts/latent_meep_search.py --latent-mode residual --n-trials 40

# Legacy σ-only
bash scripts/run_meep.sh scripts/latent_meep_search.py --latent-mode sigma --n-trials 40

# PCA subspace (8D) fit on perturb manifest
bash scripts/run_meep.sh scripts/latent_meep_search.py --latent-mode pca --pca-dim 8 --n-trials 30

# Broadband worst-case split @ 1.50, 1.55, 1.60 µm
bash scripts/run_meep.sh scripts/latent_meep_search.py --objective broadband --n-trials 20
```

---

## B2 — Surrogates

Trains four models for comparison:

- `perturb_mask_mlp` — recommended for mask ranking on perturb family  
- `perturb_latent_mlp` — matches B1/B5 search space  
- `perlin_mask_mlp` — exploratory only  
- `all_mask_mlp` — legacy mixed corpus  

```bash
python scripts/train_track_b_surrogates.py
python scripts/evaluate_surrogate_ranking.py \
  --surrogate data/phase1/track_b/surrogates/perturb_latent_mlp \
  --sim-results data/phase0/sim_results_phase0_v1_all.csv --sources perturb
```

---

## B3 — Template co-search

Searches `wg_width_um`, `arm_y_upper`, and `sigma` together (MEEP objective).

---

## B5 → MEEP verify

```bash
python scripts/surrogate_acquisition_search.py \
  --surrogate data/phase1/track_b/surrogates/perturb_latent_mlp \
  --n-proposals 1000 --top-k 30

bash scripts/run_meep.sh scripts/run_fdtd_batch.py \
  --manifest data/phase1/track_b/acquisition/top_candidates.csv \
  --output data/phase1/track_b/acquisition/meep_verify.csv --force-resim
```

---

## B6 — 70/30 splitter

```bash
bash scripts/run_meep.sh scripts/latent_meep_search.py \
  --config configs/phase2_splitter_70_30.yaml --n-trials 40
```

---

## B7 — Baselines report

```bash
python scripts/run_baselines.py --pilot   # ~15 trials each
python scripts/run_baselines.py --full    # production comparison
```

Read `data/phase1/track_b/baselines/baselines_report.json`.

---

## Success metrics (Track B gate)

| Metric | Target |
|--------|--------|
| B1 vs σ-only | Better best MEEP split at **same trial budget** |
| B2 perturb latent R² | > 0 on perturb-only holdout |
| B5 | `ranking_wins` with perturb_latent_mlp |
| B7 | Documented sim-budget curve in baselines report |

---

## Config

`configs/phase1_track_b.yaml` — single source for paths and pilot/full trial counts.
