# Tidy3D quickstart (15 FlexCredits)

## Configure API key (once)

```bash
tidy3d config migrate --delete-legacy   # once, clears ~/.tidy3d warning
```

**Never commit API keys.** Use `tidy3d configure` (writes `~/.config/tidy3d/`) or set env `SIMCLOUD_APIKEY`.

```bash
cd ~/nanophotonics-inverse-design
source .venv/bin/activate
tidy3d configure --apikey=YOUR_KEY_FROM_ACCOUNT_PAGE
python -c "import tidy3d.web as web; web.test(); print('auth OK')"
```

If a key was pasted in chat or email, **rotate it** on the account page first.

Account: https://tidy3d.simulation.cloud/account

## 2. Estimate cost before spending credits

Frugal settings: 3 champions only, coarser grid, shorter runtime.

```bash
PYTHONPATH=src .venv/bin/python scripts/crosscheck_champion_solvers.py \
  --solvers tidy3d --tidy3d-frugal --tidy3d-estimate
```

Check `data/phase1/crosscheck/crosscheck_report.json` for `estimated_flex_credits` in `runtime_note`.

## Recipe sensitivity + Tidy3D

Goal: show champions are stable under **one contracted MEEP recipe**, and Tidy3D tracks MEEP at that recipe.

MEEP sweep (local):

```bash
bash scripts/run_meep.sh scripts/study_recipe_sensitivity.py
```

See `docs/RECIPE_SENSITIVITY.md`.

## Run the three champion sims (recommended)

Skips `ref_published` (not 50/50 in-template; saves credits).

```bash
PYTHONPATH=src .venv/bin/python scripts/crosscheck_champion_solvers.py \
  --solvers tidy3d --tidy3d-frugal --append
```

## 4. Optional 4th sim

If estimates show headroom under 15 credits:

```bash
PYTHONPATH=src .venv/bin/python scripts/crosscheck_champion_solvers.py \
  --solvers tidy3d --tidy3d-frugal --tidy3d-include-ref --append
```

## Credit budget guide

| Plan | Sims | Notes |
|------|------|--------|
| **Minimal** | 3 | `local_00022`, `meep_bo_00128`, `meep_bo_00093` |
| **+ sanity** | 4 | Add `ref_published` (~0.61 split) to show solver tracks MEEP trend |

Compare results in `data/phase1/crosscheck/crosscheck_report.md` vs `meep_phase0_v1_r25`.
