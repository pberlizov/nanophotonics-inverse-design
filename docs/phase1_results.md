# Phase 1 results (in progress)

**Champion:** `local_00022` — MEEP split **0.500** @ `phase0_v1`, σ ≈ **0.014** (local refine, May 2026)

**Previous:** `meep_bo_00093` — **0.497**, σ ≈ **0.023**

---

## Week 1 — completed

| Step | Result |
|------|--------|
| v1 relabel (101 perturb) | 101/101 ok, ~103 s; mean split 0.69 |
| `ref_published` @ v1 | 0.614 (template ≠ paper 50/50) |
| MEEP search × 200 | 19/200 in-spec; best trial 93 → **0.497** (reproduced) |
| Top-10 MEEP verify | mean **0.501**, std **0.014** |
| Surrogate mask_mlp (101 rows) | R² **−0.42** — no BO |

---

## Week 2 — in progress

| Step | Status |
|------|--------|
| `run_phase1_week2.sh` | Ready (`1a` perturb skip if current → `1b` full 500 → train → ranking → local search) |
| Ranking eval (101 rows, existing surrogate) | **Done** — see below |
| Full 500 relabel + retrain | Relabel done; train fixed (160×160 perlin → pad to 180×180) |
| `meep_search_local.py` (50 trials) | **Done** — best `local_00022` split **0.500**, σ **0.014**; 4/50 in-spec |

### Ranking eval @ 101 rows (`surrogate_phase1_v1`)

| Metric | Value |
|--------|-------|
| Spearman \|err\| | **0.64** (p ≈ 7e−13) |
| `ranking_wins` | **true** (top-20 MAE 0.179 vs random 0.281) |
| In-spec in top-20 | 2 vs 0 random |
| Val R² | **−0.42** (regression still poor) |

**Interpretation:** The mask MLP ranks perturbations better than chance on v1 labels, but absolute error is unreliable. Use surrogate as a **pre-filter** (propose top-k → MEEP verify), not as the optimization objective. Surrogate-only BO stays off until R² &gt; 0 on the full v1 corpus.

**Ranking pass criterion:** `ranking_wins: true` in `data/phase1/surrogate_ranking_eval.json` (not R² alone).

---

## Inverse-design playbook (current)

1. **Sign-off:** MEEP `phase0_v1` only.  
2. **Search:** `meep_search.py` (global σ) → `meep_search_local.py` (refine).  
3. **Verify:** `run_fdtd_batch.py` on `top_candidates.csv`.  
4. **Surrogate:** proposal only until ranking wins + R² &gt; 0.

---

## Wedge A — primary strategy (active)

See [WEDGE_A.md](WEDGE_A.md). Run:

```bash
bash scripts/run_wedge_a.sh
```

## Track B — structural improvements (exploratory)

See [PHASE1_TRACK_B.md](PHASE1_TRACK_B.md). Run:

```bash
bash scripts/run_track_b.sh          # pilot
bash scripts/run_track_b.sh --full   # production-scale
```

## Deep dev

See [PHASE1_DEEP_DEV.md](PHASE1_DEEP_DEV.md). Run:

```bash
bash scripts/run_phase1_deep.sh
```

Adds multi-objective MEEP search, surrogate-ranked AL, optional `phase0_v2` calibration, GDS export.
