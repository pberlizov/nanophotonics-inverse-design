# Sprint v2 — Morph / Broadband / IL

**Status:** **ABORTED** June 2026 per user decision. Sprint jobs were stopped; existing data and logs are retained. Zenodo preprint proceeds with honest scope (split-gate pilot + diagnostic negatives).

**Prior launch:** 2026-06-08 (Zenodo/preprint upload on hold)

Prior hunts exhausted budget with **0 winners** on morph and broadband gates; IL hunt reached 6.7 dB but out of split spec. This sprint targets the three gaps with hypothesis-driven config changes.

## Prior results (baseline)

| Track | Hunt | Trials | Pass | Best metric | Blocker |
|-------|------|--------|------|-------------|---------|
| Morph Tier1 | `morph_robust_hunt_tier1` | 240 | 0 | worst=0.389 | Champion-centric, worst-first loss |
| Morph Phase2 | `phase2_morph_hunt` | 277 | 0 | worst=0.092, Δ=0.120 | `morph2_perlin_0055` close on split, fails max_delta |
| Broadband | `phase2_broadband_hunt` explore | 200 | 0 verified | obj=0.122 (`latent_meep_00182`) | No 5nm verify; champions worst≈0.36–0.49 |
| IL Phase2 | `phase2_il_hunt` | 250 | — | IL=6.7 dB | split err 0.076 (out of spec) |
| Flux audit | `flux_il_audit` | 60 sweeps | — | T≈1–3%, IL 15–19 dB | Low T is physics, not monitor placement |

Matgrid calibration artifacts exist under `data/phase1/meep_research/matgrid_calibration_*.json`.

---

## Track A — Morph robustness

**Hypothesis:** Phase2 asymmetric loss >> Tier1; failure is **max_delta_R_up** (0.12 > 0.05), not nominal split. Seed `morph2_perlin_0055`, add 15 nm stress, tighten residual (0.03), raise `weight_asymmetric`.

**Config:** `configs/sprint_morph_v2.yaml`  
**Script:** `scripts/phase2_morph_hunt.py`  
**Output:** `data/phase1/sprint_morph_v2/`

**Success gate:**
- `n_pass ≥ 1` where `morph_pass=True`
- `worst_split_error ≤ 0.05` AND `max_delta_R_up ≤ 0.05` at stress 10/15/20 nm

**Launch:**
```bash
bash scripts/run_meep.sh scripts/phase2_morph_hunt.py --config configs/sprint_morph_v2.yaml
```

**Est. runtime:** ~4–6 h (220 trials × 7 MEEP sims/trial)

---

## Track B — Broadband C-band

**Hypothesis:** `latent_meep_00182` (obj=0.122) is the best lead; champions fail flatness. Verify top-10 at 5 nm before new search; refine with `flatness_weight=0.65` and 5 nm search grid.

**Config:** `configs/sprint_broadband_v2.yaml`

### Step 1 — Verify Phase2 explore (do first)
```bash
bash scripts/run_meep.sh scripts/broadband_rescore_candidates.py \
  --candidates data/phase1/phase2_broadband_hunt/explore/top_candidates.csv \
  --wl-start 1.53 --wl-stop 1.57 --wl-step 0.005 \
  --max-worst-split-error 0.05 --limit 10 \
  --output data/phase1/sprint_broadband_v2/phase2_top10_verify.json
```

### Step 2 — Residual refine from 00182 + 00101
```bash
bash scripts/run_meep.sh scripts/broadband_refine_from_centers.py \
  --config configs/sprint_broadband_v2.yaml \
  --output-dir data/phase1/sprint_broadband_v2/refine \
  --trials-per-center 50
```

### Step 3 — Perlin explore (100 trials)
```bash
bash scripts/run_meep.sh scripts/latent_meep_search.py \
  --config configs/sprint_broadband_v2.yaml \
  --objective broadband --latent-mode perlin --n-trials 100 \
  --output-dir data/phase1/sprint_broadband_v2/explore
```

**Success gate:**
- ≥1 design with `pass_broadband_gate=True` at **wl_step=0.005** over 1.53–1.57 µm
- Target: `worst_split_error ≤ 0.05`

**Est. runtime:** verify ~1 h; refine+explore ~4–8 h

---

## Track C — IL / transmission

**Hypothesis:** Flux audit shows IL 15–19 dB stable across monitor apertures → real loss, not calibration artifact. Phase2 `weight_il=0.75` over-trades split. Use **balanced** `weight_il=0.40` + `stage2_split_penalty=25`.

**Config:** `configs/sprint_il_v2.yaml`  
**Script:** `scripts/phase2_il_hunt.py`  
**Output:** `data/phase1/sprint_il_v2/`

**Prerequisite:** Matgrid calibration complete (already in `data/phase1/meep_research/`).

**Success gate:**
- `in_spec=True` (split err ≤ 0.05) **and** `IL_db ≤ 12.0` on same design
- Stretch: IL ≤ 10 dB with split err ≤ 0.03

**Launch:**
```bash
bash scripts/run_meep.sh scripts/phase2_il_hunt.py --config configs/sprint_il_v2.yaml
```

**Est. runtime:** ~3–5 h (80 stage1 + 200 stage2 sims)

---

## CPU / launch ordering

Replication `N_REPLICATES=20` may be running (`run_release_replication.sh`). **Do not** launch all MEEP hunts simultaneously.

**Recommended stagger** (via `scripts/sprint_launch.sh stagger`):
1. **Now:** broadband verify (~1 h, low contention)
2. **+2 h:** morph sprint
3. **+4 h:** IL sprint
4. **After verify:** broadband refine+explore (manual or `sprint_launch.sh broadband`)

```bash
bash scripts/sprint_launch.sh stagger   # background with delays
# or individually:
bash scripts/sprint_launch.sh verify
bash scripts/sprint_launch.sh morph
bash scripts/sprint_launch.sh il
```

Logs: `data/phase1/release/sprint_logs/`

---

## Checklist

- [ ] Broadband verify: `phase2_top10_verify.json` — any `pass_broadband_gate`?
- [ ] Morph sprint: `sprint_morph_v2/phase2_morph_summary.json` — `n_pass ≥ 1`?
- [ ] Broadband refine: `sprint_broadband_v2/refine/broadband_refine_summary.json`
- [ ] Broadband explore: `sprint_broadband_v2/explore/latent_meep_summary.json`
- [ ] IL sprint: `sprint_il_v2/` stage1+2 — in-spec IL ≤ 12 dB?
- [ ] Update release artifacts: `bash scripts/finalize_preprint_v1.sh` (after winners)

---

## Risks

1. **CPU saturation** — replication + 3 MEEP hunts → use stagger; pause replication if needed.
2. **Morph max_delta** — may need `morph_robust_search.py` with `weight_max_delta=3` if asymmetric hunt stalls.
3. **Broadband** — 0.122 coarse objective may still fail 5 nm gate (dispersion).
4. **IL** — true T~1–3% may be fundamental at r25; IL≤12 dB may require geometry/port recipe change, not search alone.
