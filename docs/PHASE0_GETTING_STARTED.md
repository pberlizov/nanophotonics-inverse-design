# Phase 0 — Getting started

**Goal:** Prove **decode → score → verify** on one device before custom models.

**Schedule:** Target **7–10 days** (aggressive). Conservative **4-week** breakdown is in [ROADMAP.md](ROADMAP.md).

**Install:** [INSTALL.md](INSTALL.md) — **`drcgenerator` requires Python `3.12.12` exactly**, not 3.11 or generic 3.12.

---

## Wedge (locked)

| Choice | Decision |
|--------|----------|
| Device | 50/50 power splitter |
| Platform | EBL (`EBeamModel`) |
| Band | 1.5–1.6 µm |
| Manifold | Reuse `drcgenerator` |
| Search | Latent BO (not RL first) |
| Active learning | After Phase 0 gate |

Config: [configs/phase0.yaml](../configs/phase0.yaml)

---

## Day 1 — Full setup commands

**Easiest:** `bash scripts/setup.sh` then `source .venv/bin/activate`. Details: [INSTALL.md](INSTALL.md).

```bash
cd ~/nanophotonics-inverse-design
bash scripts/setup.sh
source .venv/bin/activate
which python          # .../nanophotonics-inverse-design/.venv/bin/python
python --version      # Python 3.12.12
```

Manual equivalent (do **not** use system `pip` — it is often Python 3.14):

```bash
cd ~/nanophotonics-inverse-design
rm -rf .venv
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12.12
uv venv --python 3.12.12 .venv
mkdir -p external
git clone https://github.com/Photonic-Architecture-Laboratories/drcgenerator.git external/drcgenerator
uv pip install -e external/drcgenerator --python .venv/bin/python
.venv/bin/python -c "import jax; import drcgenerator; print('ok')"
```

**pyenv alternative** (if you prefer):

```bash
brew install pyenv
pyenv install 3.12.12
cd ~/nanophotonics-inverse-design
rm -rf .venv
pyenv local 3.12.12
python --version
python -m venv .venv
source .venv/bin/activate
# then same git clone + pip install -e external/drcgenerator
```

**First decode check** (optional CLI smoke test after install):

```bash
source ~/nanophotonics-inverse-design/.venv/bin/activate
python <<'PY'
import jax
import jax.numpy as jnp
from drcgenerator.models import EBeamModel
from drcgenerator import ebeam_ps_latent_space_ex_path

model = EBeamModel()
z = jnp.load(ebeam_ps_latent_space_ex_path)
rng = jax.random.PRNGKey(0)
params = model.init(rng, z)
out = model.apply(params, z)
print("decode shape:", getattr(out, "shape", type(out)))
PY
```

**Notebook:** `external/drcgenerator/examples/e_beam_inference.ipynb`

---

## Sprint plan (target 7–10 days)

### Sprint 0 — Setup (Day 1) — done

- [x] Python 3.12.12 venv per [INSTALL.md](INSTALL.md)
- [x] `drcgenerator` imports
- [x] `scripts/verify_setup.py` — EBL decode OK

### Sprint 1 — Decode batch (Days 2–3) — **you are here**

```bash
source ~/nanophotonics-inverse-design/.venv/bin/activate
uv pip install -r requirements-phase0.txt --python .venv/bin/python
python scripts/decode_batch.py --n-samples 500 --preview-png
```

- [ ] Full batch (500+) → `data/phase0/manifest.csv`
- [ ] Review `data/phase0/drc_report.json` and `data/phase0/previews/`
- **Exit:** pass rate documented; masks ready for FDTD subsample

### Sprint 2 — MEEP labels (Days 2–5) — **you are here**

Install MEEP: [MEEP_SETUP.md](MEEP_SETUP.md). Recipe: [sim_recipe_phase0.md](sim_recipe_phase0.md).

```bash
conda activate mp
cd ~/nanophotonics-inverse-design
pip install pandas pyyaml tqdm numpy   # if needed in mp env

# Pilot (fast, low resolution)
python scripts/run_fdtd_batch.py --limit 10 --resolution 15 --verbose

# Production batch (~500 masks, slower)
python scripts/run_fdtd_batch.py --resolution 25
```

- [ ] Pilot 10 sims succeed (`status=ok` in `data/phase0/sim_results.csv`)
- [ ] Full batch (or 200+ stratified subsample)
- **Exit:** `split_ratio_upper` column populated for surrogate training

### Sprint 3 — Surrogate + search (Days 5–7)

- [x] `scripts/train_surrogate.py` (latent_mlp / mask_mlp / mask_cnn; `--sources`)
- [x] `scripts/latent_search.py` (BO over `z`)
- [x] FDTD verify top-20 vs random (see [phase0_results.md](phase0_results.md))
- **Exit:** Pipeline yes; surrogate calibration no — conditional go

### Sprint 4 — Gate (Day 8–10)

- [x] Baselines in [phase0_results.md](phase0_results.md)
- [x] `scripts/evaluate_phase0.py` → `data/phase0/gate_metrics.json`
- [x] **Conditional go** → Phase 1 (mask surrogate + active learning)

---

## Conservative week map (reference only)

| Week | Original bucket | Maps to sprint |
|------|-----------------|----------------|
| 1 | Manifold decode | Sprint 0–1 |
| 2 | FDTD labels | Sprint 2 |
| 3 | Surrogate + search | Sprint 3 |
| 4 | Gate | Sprint 4 |

---

## Open questions

| ID | Question | Default |
|----|----------|---------|
| Q1 | Simulator? | Tidy3D if licensed; else MEEP |
| Q2 | Relax `==3.12.12` pin? | No — use uv/pyenv (see INSTALL dev hatch) |

---

## After Phase 0

[ROADMAP.md](ROADMAP.md) Phase 1 — own manifold, ensemble surrogate, active learning.
