# Improvement backlog (from replication + novelty + surrogate)

**Last updated:** 2026-06-03  
**Replication results:** [SIM_BUDGET_REPLICATION_RESULTS.md](SIM_BUDGET_REPLICATION_RESULTS.md)

Status: `[ ]` todo · `[~]` partial · `[x]` done

---

## Three parallel tracks (avoid thrash)

The backlog is growing because we now have **three independent goals**, not one linear queue:

| Track | Goal | Gate | Do not block on |
|-------|------|------|-----------------|
| **A — Product contract** | MEEP `phase0_v1` mesh-stable, promotable recipe | Triple-pass on 3 champions | invrs-gym scores |
| **B — Danis Path A** | JAX grad through G(z) + diff EM | One champion grad step beats BO baseline | MEEP mesh audit |
| **C — External credibility** | invrs-gym Ceviche leaderboard comparison | Baseline + one optimized entry | Our MEEP template |

**Current focus order:** MEEP-gated AL loop + matgrid → validation replicate → ensemble ranker; lead **`cand_000261`**.

**Lead:** `cand_000261` (prod 0.495, promotable) — 6/6 champion `sdf_geom` gate; see `promotion_validation.md`.

**Explicit defer:** fmmax (RCWA — wrong physics for finite splitter), Path B surrogate grad, Tidy3D invdes until Ceviche gym path is wired, generator fine-tune (D3) until D1–D2 pass.

---

## Tier 0 — Biggest wins (implement first)

| Status | Item | Rationale |
|--------|------|-----------|
| [x] | **Merge 5× sim-budget MEEP into corpus** | ~5k σ-local labels in the search distribution; replication MEEP was unused for training |
| [x] | **Retrain ranker on expanded corpus** (`surrogate_improved`) | Direct R² + ranking lift from label count |
| [x] | **`hierarchical_35` as expert baseline** | Beats 50/65 at B=100 in replication |
| [x] | **Champion-centered search** | Centers: ref + `local_00022` + `meep_bo_00128` latents |
| [x] | **σ as mask_mlp input feature** | Labels are σ-structured; cheap R²/ranking gain |
| [~] | **Near-target training filter** (optional `max_abs_err`) | Tried `0.15` — **hurt** val R² (−4.7); **disabled** in `wedge_a_improved.yaml` |

**Commands:**

```bash
bash scripts/run_post_replication_improvements.sh
# or stepwise:
python scripts/merge_search_labels_into_corpus.py --config configs/corpus_merge.yaml
python scripts/train_wedge_a_surrogate.py --config configs/wedge_a_improved.yaml
python scripts/evaluate_surrogate_ranking.py --surrogate data/phase1/wedge_a/surrogate_improved ...
```

---

## Tier 1 — High leverage (implemented or wired)

| Status | Item | Notes |
|--------|------|-------|
| [x] | **Diverse surrogate shortlist** | Greedy Hamming on pooled masks — `diverse_top_k` in candidate pool |
| [x] | **Val Spearman / NDCG@k in metrics.json** | Better than R² alone for ranker gate |
| [x] | **AL rounds 6–10** after merge | More σ-band labels; run when MEEP budget available |
| [x] | **Refresh pilot bundle** | `bash scripts/run_pilot.sh --report-only` (2026-05-26) |
| [x] | **Validation replicate run_06** | `bash scripts/run_validation_replicate.sh` (B=100, improved ranker) |

---

## Tier 1.5 — Differentiable inverse design (Danis et al. 2602.03142)

**Strategy:** Hybrid BO (global) + grad refinement (local). See [DIFFERENTIABLE_INVERSE_DESIGN.md](DIFFERENTIABLE_INVERSE_DESIGN.md).

| Status | Item | Notes |
|--------|------|-------|
| [x] | **Phase D0 — analytical MEEP geometry** | **`sdf_geom` triple-pass** — 0.500/0.500 r25 & r50 on all 3 champions; see `d0_geometry_report.md` |
| [x] | **Phase D1 — JAX grad through drcgenerator** | `decode_soft` + `scripts/spike_jax_decode_grad.py` |
| [x] | **Phase D2 — JAX diff EM proxy** | `invrs_adapter.latent_to_gym_params` + `gym_loss_and_metric`; gym-native opt `optimize_invrs_ceviche.py` (~0.085) |
| [ ] | **Phase D3 — Fine-tune G on corpus** | L_topo + recon vs frozen decodes |
| [x] | **Phase D4 — `refine_champion_grad.py`** | Gym soft in_spec (~0.01–0.02); **MEEP verify** (`verify_refined_champions.md`): prod r25 OK only on `local_00022` (Δ+0.02); `meep_bo_*` → ~0.97–0.99 — **do not promote refined z** |
| [~] | **Surrogate refine + prod MEEP verify** | `refine_champion_surrogate.py` (STE + trust region) + `refine_champion_product.py`; gate on **prod r25** only. Gym refine (`refine_champion_grad.py`) **exploratory only**. |
| [ ] | **Path B surrogate grad** | Defer — R² ≈ −0.6 (ranking wins; refine uses pred objective not R²) |

---

## Tier 1.6 — invrs-gym external benchmark (Track C)

**Repos:** [invrs-gym](https://github.com/invrs-io/gym), [leaderboard](https://github.com/invrs-io/leaderboard). JAX-native; Ceviche FDFD inside.

| Status | Item | Notes |
|--------|------|-------|
| [x] | **`invrs-gym` in venv** | `uv pip install invrs-gym`; pulls `ceviche`, `fmmax` |
| [x] | **Baseline eval spike** | `benchmark_invrs_ceviche.py` — random init eval_metric −3.45 |
| [x] | **Gym-native topology opt** | 80-step L-BFGS → **in_spec** (eval_metric **0.085**); `optimize_invrs_ceviche.py` |
| [x] | **Leaderboard comparison table** | `compare_invrs_leaderboard.py` → `leaderboard_comparison.md` (top ~**0.009**) |
| [x] | **Wire drcgenerator → gym density** | `latent_to_gym_params` + `refine_champion_grad.py` (grad in **z**, not pasted masks) |
| [ ] | **fmmax** | Defer — metagrating/diffractive challenges only |

**Not the same problem as MEEP `phase0_v1`:** gym Ceviche power splitter is 1.6×1.6 µm, 400 nm wg, FDFD — comparable *class*, different template.

---

## Tier 2 — Product / science

| Status | Item | Notes |
|--------|------|-------|
| [~] | **MEEP research arc** | Per-champion matgrid sweeps running; see `scripts/run_matgrid_backlog.sh` |
| [x] | **Promote mesh-stable recipe** | **`phase0_v1_sdf_geom`** promoted → `configs/phase0.yaml` |
| [ ] | **Dual-track pitch** | σ-ball ranker vs Perlin exploration (separate stories) |
| [ ] | **Perlin-only ranker** | Only if searching Perlin with ML |
| [ ] | **Client MEEP calibration** | Paid change order |
| [ ] | **IL hard gate in search** | `search.weight_il` already in config |
| [x] | **Pairwise / rank loss** | `round_rank` MEEP: **cand_000261** prod 0.4946 (\|err\| 0.0054) beats regression round5; see `meep_gated_shortlist/round_rank/vs_regression_round5.md` |

---

## Tier 3 — Defer / do not pursue

| Item | Why |
|------|-----|
| Surrogate-only BO | Failed thesis |
| Sell val R² to customers | Negative holdout; ranking wins |
| Mixed perlin+perturb single ranker for σ-search | Hurts σ ranking |
| `abs_split_error` as primary target | Worse R² in our tests |
| Lead with B=30 sim-budget | All policies weak |

---

## R² improvement track (separate from product gate)

| Status | Tactic | Expected |
|--------|--------|----------|
| [x] | More σ-local labels (replicate merge) | Largest R² lift |
| [~] | Near-target subset / emphasis | **Regressed** R² at 0.15; disabled |
| [x] | σ feature on mask_mlp | Moderate R² + ranking |
| [ ] | 500–1000+ labels + AL 6–8 | Positive R² possible |
| [x] | Rank-aware loss (pairwise_rank) | MEEP-gated `round_rank` validates ranking on truth; prod lead **cand_000261** |
| [ ] | Ensemble 3–5 seeds | Stability |

**Gate for release:** `ranking_wins` + sim-budget B=100 — **not** val R² > 0.

---

## Verification after improvements (2026-05-26 run)

| Model | n_train | val_r2 | val_spearman_abs_err | ranking_wins @ top-20 (n=3191) |
|-------|---------|--------|----------------------|--------------------------------|
| `surrogate_production` (301 labels) | 240 | −0.36 | — | true (tied 3 in-spec on merged eval) |
| `surrogate_improved` + near-target 0.15 | 964 | −4.72 | 0.13 | **false** |
| **`surrogate_improved`** (merged + σ, full set) | 2552 | −0.79 | 0.36 | **true** (8 vs 3 in-spec) |

Corpus: **3622** ok rows in `sim_results_phase0_v1_all.csv` (+2890 from 5× sim-budget replicates).

```bash
cat data/phase1/wedge_a/surrogate_improved/metrics.json
cat data/phase1/wedge_a/ranking_eval_improved.json
wc -l data/phase0/sim_results_phase0_v1_all.csv
```
