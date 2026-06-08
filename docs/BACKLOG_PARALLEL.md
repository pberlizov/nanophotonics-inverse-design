# Parallel backlog (while MEEP matgrid runs)

**MEEP jobs (do not start more):** `calibrate_matgrid_geometry.py` on `local_00022` (~28% → then `meep_bo_00093`).

**Safe in parallel (JAX / CPU, `.venv`):**

| Priority | Track | Script | Est. time |
|----------|-------|--------|-----------|
| 1 | C | `scripts/optimize_invrs_ceviche.py` | 10–30 min (lightweight, 50 steps) |
| 2 | C | `scripts/compare_invrs_leaderboard.py` | <1 min (network) |
| 3 | B | `scripts/spike_jax_decode_grad.py` | <1 min |
| 4 | B | `scripts/benchmark_invrs_ceviche.py` | done |
| — | A | MEEP crosscheck with `sdf_geom` | **after** matgrid |

```bash
cd ~/nanophotonics-inverse-design
source .venv/bin/activate

# Track C — gym optimization (JAX only)
PYTHONPATH=src python scripts/optimize_invrs_ceviche.py --steps 50

# Track C — leaderboard table (density + refine_grad, best eval_metric)
PYTHONPATH=src python scripts/compare_invrs_leaderboard.py

# Track C — champion latent refine (261, 003, …) with patience / best metric
PYTHONPATH=src python scripts/refine_champion_grad.py
# or one champion:
PYTHONPATH=src python scripts/refine_champion_grad.py --sample-id cand_000261

# Track B — D1 grad through drcgenerator
PYTHONPATH=src python scripts/spike_jax_decode_grad.py
```
