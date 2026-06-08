# MEEP setup (Sprint 2)

## Recovery: `miniforge3` exists but `conda.sh` missing

This means a **partial install** (installer stopped mid-way). Pick one:

### A — Finish install with `-u` (try first)

```bash
curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o /tmp/mf.sh
bash /tmp/mf.sh -u -b -p ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh
bash ~/nanophotonics-inverse-design/scripts/install_meep.sh
```

### B — Clean reinstall

```bash
rm -rf ~/miniforge3
bash ~/nanophotonics-inverse-design/scripts/install_meep.sh
```

### C — Use micromamba only (if `~/miniforge3/micromamba` exists)

```bash
export MAMBA_ROOT_PREFIX=~/miniforge3
~/miniforge3/micromamba create -n mp -c conda-forge pymeep python=3.11 -y
~/miniforge3/micromamba run -n mp python -m pip install pandas pyyaml tqdm numpy
bash ~/nanophotonics-inverse-design/scripts/run_meep.sh scripts/run_fdtd_batch.py --limit 10 --resolution 15
```

(`run_meep.sh` will auto-`pip install` those packages if pandas is missing.)

`run_meep.sh` uses `micromamba run -n mp` — **no `conda activate`**, works even without `conda.sh`.

---

## Why you see `No module named 'meep'`

| Environment | Has `drcgenerator`? | Has `meep`? |
|-------------|---------------------|-------------|
| `.venv` (Python **3.12.12**) | Yes | **No** — not on PyPI for this stack |
| Conda env **`mp`** | No (not needed) | **Yes** |

If you ran:

```bash
source .venv/bin/activate
python scripts/run_fdtd_batch.py   # WRONG env
```

you will always get `ModuleNotFoundError`. Use **`conda activate mp`** instead.

---

## One-command install (no conda yet)

From Terminal:

```bash
cd ~/nanophotonics-inverse-design
bash scripts/install_meep.sh
```

This installs **Miniforge** to `~/miniforge3` and creates env **`mp`** with `pymeep`.

Add to `~/.zshrc` (once):

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
```

Open a **new** terminal tab, then:

```bash
conda activate mp
python -c "import meep; print('meep ok')"
```

---

## Run FDTD batch (correct env)

```bash
conda activate mp
cd ~/nanophotonics-inverse-design
python -m pip install pandas pyyaml tqdm numpy   # once per mp env

# Pilot
python scripts/run_fdtd_batch.py --limit 10 --resolution 15 --verbose

# Full run
python scripts/run_fdtd_batch.py --resolution 25
```

Confirm you are not on `.venv`:

```bash
which python
# should be: .../miniforge3/envs/mp/bin/python
# NOT: .../nanophotonics-inverse-design/.venv/bin/python
```

---

## Manual install (if script fails)

### Apple Silicon (try native arm64 first)

```bash
curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o /tmp/mf.sh
bash /tmp/mf.sh -b -p ~/miniforge3
source ~/miniforge3/etc/profile.d/conda.sh
conda create -n mp -c conda-forge pymeep python=3.11 -y
conda activate mp
python -c "import meep; print('ok')"
```

### Apple Silicon fallback (Rosetta / x86_64)

```bash
CONDA_SUBDIR=osx-64 conda create -n mp -c conda-forge pymeep python=3.11 -y
conda activate mp
conda config --env --set subdir osx-64
python -c "import meep; print('ok')"
```

---

## Two-env workflow (summary)

```bash
# Decode (Component 1)
source ~/nanophotonics-inverse-design/.venv/bin/activate
python scripts/decode_batch.py --n-samples 500

# MEEP labels (Component 2 ground truth)
conda activate mp
python scripts/run_fdtd_batch.py --resolution 25
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No module named 'meep'` | Wrong Python — use `conda activate mp`, not `.venv` |
| `conda: command not found` | Run `bash scripts/install_meep.sh` or install Miniforge manually |
| Script says "must use MEEP conda env" | You activated `.venv`; `conda activate mp` |
| Very slow sims | `--resolution 15` for pilots |
