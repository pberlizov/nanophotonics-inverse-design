# Phase 1 — Deep development

**Prerequisite:** Week 2 complete (`sim_results_phase0_v1_all.csv`, champion `local_00022`, `ranking_wins: true`).

This track adds **multi-objective MEEP search**, **surrogate-ranked active learning** (MEEP verifies only top-k), **template v2 spike**, and **layout export**.

---

## Tracks

| Track | Script / config | Purpose |
|-------|-----------------|--------|
| **B — Search++** | `meep_search.py --objective multi` | Split + insertion-loss penalty |
| **C — Surrogate AL** | `surrogate_ranked_al_round.py` | 800 surrogate proposals → MEEP ×12 → retrain |
| **D — Fab export** | `export_layout_gds.py` | GDS from champion mask |
| **E — Template v2** | `phase0_v2` in `meep_sim.py` | Port/arm tweak; run `calibrate_meep.py` before trusting |

---

## One-shot pipeline

```bash
cd ~/nanophotonics-inverse-design
source .venv/bin/activate
chmod +x scripts/run_phase1_deep.sh

# Optional GDS support
uv pip install gdstk --python .venv/bin/python

bash scripts/run_phase1_deep.sh
```

**Runtime (rough):** export &lt;1 min; calibrate v2 ~1 min; global search 150 × ~3 s ≈ **8 min**; local 50 ≈ **3 min**; AL presearch ~2 min + MEEP verify 12 ≈ **1 min** + retrain ~10 s.

---

## Step-by-step (manual)

### Multi-objective global search

```bash
bash scripts/run_meep.sh scripts/meep_search.py \
  --config configs/phase1.yaml \
  --output-dir data/phase1/meep_search_deep \
  --n-trials 150 \
  --objective multi \
  --sigma-min 0.008 --sigma-max 0.035
```

Loss = `|split − 0.5| + 0.15 × max(0, IL − 12 dB)` (see `search_objectives.meep_search_loss`).

### Surrogate-ranked AL (round 1)

```bash
python scripts/surrogate_ranked_presearch.py \
  --surrogate data/phase1/surrogate_mask_v1_full \
  --n-proposals 800 --top-k 40

bash scripts/run_meep.sh scripts/run_fdtd_batch.py \
  --manifest data/phase1/al_round_01/meep_verify.csv \
  --output data/phase1/al_round_01/meep_verify_results.csv \
  --force-resim --no-skip-existing

# Or orchestrated:
python scripts/surrogate_ranked_al_round.py --round 1
```

**Gate:** `data/phase1/surrogate_ranking_eval.json` must have `ranking_wins: true` (or pass `--force`).

### Template v2 calibration

```bash
bash scripts/run_meep.sh scripts/calibrate_meep.py \
  --config configs/phase1.yaml --recipe-version phase0_v2 --verbose
```

Only switch production labels to `phase0_v2` after G1a (empty/full ≈ 0.5) still passes.

### Layout export

```bash
python scripts/export_layout_gds.py \
  --sample-id local_00022 \
  --mask data/phase1/meep_search_local/candidates/masks/local_00022_mask.npy \
  --pitch-um 4.0
```

---

## Decision rules (unchanged)

1. **Design sign-off:** MEEP `phase0_v1` until v2 calibrates.  
2. **Promotion:** Every surrogate shortlist → MEEP verify before fab.  
3. **No surrogate-only BO** until holdout R² &gt; 0 on v1 corpus.  
4. **Champion:** Best *confirmed* MEEP split in-spec; currently `local_00022`.

---

## Artifacts

| Path | Content |
|------|---------|
| `configs/phase1.yaml` | Champion, search weights, AL sizes |
| `data/phase1/meep_search_deep/` | Multi-objective global search |
| `data/phase1/al_round_01/` | Presearch + verify + retrained surrogate |
| `data/phase1/exports/` | PNG + GDS + metadata |
