# Contributing

Thank you for interest in this research benchmark. This is an early-stage open release tied to a Zenodo preprint — issues and small fixes are welcome; large feature PRs are best discussed first.

## Before you start

- **Python:** 3.12.12 (required by `drcgenerator`) — see [docs/INSTALL.md](docs/INSTALL.md)
- **MEEP:** micromamba env `mp` via `bash scripts/install_meep.sh`
- **Scope:** simulation-only under frozen `phase0_v1`; MEEP is the promotion authority

## Development setup

```bash
bash scripts/setup.sh
bash scripts/install_meep.sh
.venv/bin/python scripts/verify_setup.py
```

## Running checks

```bash
.venv/bin/python -m pytest tests/ -q
bash scripts/finalize_preprint_v1.sh   # figures + PDF + readiness (CPU-only parts)
.venv/bin/python scripts/check_preprint_v1_readiness.py
```

MEEP jobs are long-running. Use `bash scripts/run_meep.sh <script>` so the `mp` env is active.

## Reporting issues

Include:

1. Recipe / config YAML path
2. Command line used
3. Whether failure is in surrogate (CPU) or MEEP (FDTD)
4. Relevant log tail from `data/phase1/release/` or terminal

## Claim hygiene

Please keep README, docs, and PR descriptions aligned with [docs/OPEN_SOURCE_RELEASE.md](docs/OPEN_SOURCE_RELEASE.md):

- **In scope:** verified split-gate yield, sim-budget accounting, reproducible pipeline
- **Out of scope:** deployment-ready devices, calibrated IL, C-band winners, morph-robust champions

## Citation

See [CITATION.cff](CITATION.cff) and the Zenodo DOI once uploaded.
