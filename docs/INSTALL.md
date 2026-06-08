# Installation — Python 3.12.12 + drcgenerator

`drcgenerator` pins **`requires-python = "==3.12.12"`** exactly (see `external/drcgenerator/pyproject.toml`).

You will see errors like:

```text
ERROR: Package 'drcgenerator' requires a different Python: 3.14.0 not in '==3.12.12'
ERROR: Package 'drcgenerator' requires a different Python: 3.11.14 not in '==3.12.12'
```

**Cause:** `pip` or `python` on your PATH is **not** the project venv’s **3.12.12** interpreter (macOS default is often 3.14; an old `.venv` may be 3.11).

**Fix:** Delete `.venv`, recreate with 3.12.12, install with **`uv pip --python .venv/bin/python`** (see below). Never run bare `pip install` until `which python` shows `.venv/bin/python` and `python --version` is **3.12.12**.

---

## One-command setup (recommended)

```bash
cd ~/nanophotonics-inverse-design
bash scripts/setup.sh
source .venv/bin/activate
which python
python --version
```

Expected:

```text
.../nanophotonics-inverse-design/.venv/bin/python
Python 3.12.12
```

---

## Method A — `uv` (manual steps)

Install [uv](https://docs.astral.sh/uv/) if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# restart shell or: source $HOME/.local/bin/env
```

Full project setup:

```bash
cd ~/nanophotonics-inverse-design

# Remove any old venv built with the wrong Python
rm -rf .venv

# Install and use exactly 3.12.12
uv python install 3.12.12
uv venv --python 3.12.12
source .venv/bin/activate

# Confirm version (must print 3.12.12)
python --version

# Clone manifold reference (if not already present)
mkdir -p external
git clone https://github.com/Photonic-Architecture-Laboratories/drcgenerator.git external/drcgenerator

# Install drcgenerator — MUST use uv pip with explicit interpreter (uv venv has no pip binary by default)
uv pip install -e external/drcgenerator --python .venv/bin/python

# Verify (always use .venv/bin/python if unsure)
.venv/bin/python -c "import jax; import drcgenerator; print('jax', jax.__version__, 'drcgenerator ok')"
```

**After `source .venv/bin/activate`:** use `python` and `uv pip`, not system `pip`. If `pip install` still fails, you are on the wrong Python — run `which python` and `python --version`.

---

## Method B — `pyenv`

```bash
brew install pyenv
pyenv install 3.12.12
cd ~/nanophotonics-inverse-design
pyenv local 3.12.12

rm -rf .venv
python --version    # must show 3.12.12
python -m venv .venv
source .venv/bin/activate

mkdir -p external
git clone https://github.com/Photonic-Architecture-Laboratories/drcgenerator.git external/drcgenerator
pip install -U pip
pip install -e external/drcgenerator
python -c "import drcgenerator; print('ok')"
```

---

## Method C — Homebrew `python@3.12` (usually **not** sufficient)

Homebrew ships **3.12.x** but rarely **3.12.12** exactly. Only use this if:

```bash
/opt/homebrew/opt/python@3.12/bin/python3.12 --version
```

prints **`Python 3.12.12`**. Otherwise use Method A or B.

```bash
brew install python@3.12
cd ~/nanophotonics-inverse-design
rm -rf .venv
/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv .venv
source .venv/bin/activate
python --version
# continue with clone + pip install -e external/drcgenerator as above
```

---

## Optional — Jupyter for upstream examples

```bash
source ~/nanophotonics-inverse-design/.venv/bin/activate
pip install jupyter ipykernel
python -m ipykernel install --user --name nanophotonics-phase0 --display-name "nanophotonics (3.12.12)"
cd external/drcgenerator/examples
jupyter notebook e_beam_inference.ipynb
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `3.14.0 not in '==3.12.12'` | You used system/Python 3.14 `pip`. Run `bash scripts/setup.sh` or `uv pip install ... --python .venv/bin/python` |
| `3.11.14 not in '==3.12.12'` | Old `.venv`; `rm -rf .venv` and rerun setup |
| `not in '==3.12.12'` (generic) | `rm -rf .venv`; only create with `uv venv --python 3.12.12` |
| `pip: command not found` in venv | Normal for `uv venv`; use `uv pip` or `.venv/bin/python -m pip` after `uv pip install pip` |
| Wrong `python` after activate | `which python` → `.../nanophotonics-inverse-design/.venv/bin/python` |
| JAX fails on Apple Silicon | Use 3.12.12 venv only; JAX 0.7.2 has arm64 wheels |
| `git clone` hooks error | Clone in Terminal.app: `GIT_TEMPLATE_DIR= git clone ...` or ignore if repo files exist |

---

## Dev-only escape hatch (not for production)

If you cannot install 3.12.12 and need a quick local test, temporarily relax `requires-python` in `external/drcgenerator/pyproject.toml` from `==3.12.12` to `>=3.12,<3.13`, then reinstall. **Revert before citing reproducibility** — upstream pins exact versions for JAX/Flax compatibility.
