# Nanophotonics Inverse Design

**Reproducible MEEP-gated search benchmark for DRC-feasible 1×2 power splitters.**

Open research release: frozen forward model (`phase0_v1`), sim-budget replication, documented negative results (broadband / morph / IL), and scripts to extend the **manifold → rank → MEEP verify → corpus** pattern. Simulation-only — not deployment-ready PIC IP.

| | |
|---|---|
| **Zenodo** | DOI TBD — [upload checklist](docs/ZENODO_RELEASE.md) · tag `v1.0-preprint` |
| **License** | [MIT](LICENSE) |
| **Python** | 3.12.12 (`.venv`) |
| **MEEP** | conda env `mp` — [setup](docs/MEEP_SETUP.md) |
| **Preprint** | [docs/preprint/manuscript.pdf](docs/preprint/manuscript.pdf) |
| **Cite** | [CITATION.cff](CITATION.cff) |
| **README verified** | **2026-06-08** |

**Start here:** [docs/OPEN_SOURCE_RELEASE.md](docs/OPEN_SOURCE_RELEASE.md) · [docs/ADOPTERS.md](docs/ADOPTERS.md) · [sim recipe](docs/sim_recipe_phase0.md)

---

## Quick start

```bash
git clone https://github.com/pberlizov/nanophotonics-inverse-design.git
cd nanophotonics-inverse-design
bash scripts/setup.sh
source .venv/bin/activate
python scripts/verify_setup.py

# Refresh preprint artifacts, PDF, Zenodo bundle (CPU-only; safe while MEEP runs elsewhere)
bash scripts/finalize_preprint_v1.sh

# Read the preprint
open docs/preprint/manuscript.pdf   # macOS; or xdg-open / evince on Linux
```

MEEP verification requires `conda activate mp` — always invoke via `bash scripts/run_meep.sh …`. See [docs/INSTALL.md](docs/INSTALL.md).

---

## What we proved / what we didn't

| | In scope (documented) | Out of scope (do not claim) |
|---|------------------------|-----------------------------|
| **Primary result** | MEEP-gated surrogate search improves **verified 50/50 split yield per sim dollar** under frozen `phase0_v1` | “Fewer MEEP calls” as lead without budget context |
| **Pipeline** | Reproducible decode → rank → MEEP verify → corpus; `repro_manifest.json` | Surrogate-only sign-off |
| **Negative science** | Broadband hunt (0 flat winners), morph stress sensitivity, IL/split tradeoffs | C-band WDM, morph robustness, low IL |
| **Sim budget** | Six-seed pilot: `surrogate_rank` **15.0 ± 3.2** in-spec vs **10.6 ± 4.8** σ-only at B=100 | Definitive n=20 policy ordering |
| **Validation** | Simulation-only 2D TE MEEP | Foundry DRC, fab yield, calibrated IL |

Full claim contract: [docs/OPEN_SOURCE_RELEASE.md](docs/OPEN_SOURCE_RELEASE.md).

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [Architecture](#2-architecture)
3. [Installation](#3-installation)
4. [Repository layout](#4-repository-layout)
5. [Verified results](#5-verified-results)
6. [Implementation status](#6-implementation-status)
7. [Limitations](#7-limitations)
8. [Re-verify locally](#8-re-verify-locally)
9. [Command reference](#9-command-reference)
10. [Documentation index](#10-documentation-index)
11. [Contributing & release](#11-contributing--release)

---

## 1. Executive summary

This repo implements **MEEP-native inverse design on a DRC-feasible photonic manifold** (`external/drcgenerator`):

1. **Search** on-manifold (σ-perturbation, Perlin, MEEP-driven BO).
2. **Optionally pre-filter** with a surrogate (ranking only — not sign-off).
3. **Verify** every promoted design with **MEEP** (`phase0_v1`).
4. **Active learning** — append labels, retrain ranker.

### Headline findings (MEEP-verified)

| Finding | Source |
|---------|--------|
| `ref_published` → split **0.614** in our MEEP template (not 50/50) | `data/phase0/calibration_phase0_v1.json` |
| Champion `local_00022` → **0.500** split at **0.47%** pixel Hamming from ref | `data/phase1/novelty/novelty_report.json` |
| At **B=100**, surrogate pre-filter **15.0 ± 3.2** in-spec vs **10.6 ± 4.8** (σ-only), six seeds | [docs/SIM_BUDGET_REPLICATION_RESULTS.md](docs/SIM_BUDGET_REPLICATION_RESULTS.md) |
| Surrogate: `ranking_wins: true`; **val R² < 0** — pre-filter only | `data/phase1/wedge_a/ranking_eval.json` |

**Surrogate (honest):** pitch **MEEP search + optional pre-filter**, not ML regression.

---

## 2. Architecture

```mermaid
flowchart TB
  subgraph cheap [Cheap loop — .venv]
    spec[Target spec YAML]
    gen[Latent perturb + decode]
    drc[Heuristic DRC]
    sur[Surrogate rank]
    spec --> gen --> drc --> sur
  end
  subgraph expensive [Expensive loop — conda mp]
    meep[MEEP verify phase0_v1]
    corpus[(sim_results CSV)]
    sur -->|top-k| meep --> corpus
  end
  corpus -->|retrain| sur
```

| Component | Module / scripts | Status |
|-----------|------------------|--------|
| Manifold decode | `nano_inv.manifold`, `decode_batch.py` | Done |
| Heuristic DRC | `nano_inv.drc_heuristic` | Done (not foundry DRC) |
| MEEP forward model | `nano_inv.meep_sim`, `run_fdtd_batch.py` | Done — `phase0_v1` |
| Surrogate ranker | `train_wedge_a_surrogate.py`, `evaluate_surrogate_ranking.py` | Done |
| Sim-budget study | `run_sim_budget_study.py`, 6× replicate | Done |
| Active learning | `run_wedge_a_round.py` | Done — round 1 |
| Fab validation | — | Not started |

Technical wedge reference (historical product framing archived): [docs/WEDGE_A.md](docs/WEDGE_A.md).

---

## 3. Installation

Two environments:

| Env | Purpose | Activate |
|-----|---------|----------|
| **`.venv`** (Python **3.12.12**) | Decode, ML, reporting | `source .venv/bin/activate` |
| **`mp`** (conda) | MEEP FDTD | `bash scripts/run_meep.sh …` |

```bash
bash scripts/setup.sh
source .venv/bin/activate
python scripts/verify_setup.py
```

| Doc | Content |
|-----|---------|
| [docs/INSTALL.md](docs/INSTALL.md) | Python + `drcgenerator` |
| [docs/MEEP_SETUP.md](docs/MEEP_SETUP.md) | Conda MEEP env `mp` |
| [docs/sim_recipe_phase0.md](docs/sim_recipe_phase0.md) | Frozen forward model |

---

## 4. Repository layout

```
nanophotonics-inverse-design/
├── README.md
├── LICENSE · CITATION.cff · CONTRIBUTING.md
├── configs/                  phase0, wedge_a, pilot YAMLs
├── src/nano_inv/               Core library
├── scripts/                    CLI orchestrators
├── external/drcgenerator/      Manifold decode (submodule)
├── data/                       MEEP labels, release artifacts (gitignored)
├── docs/
│   ├── OPEN_SOURCE_RELEASE.md  Public framing
│   ├── ADOPTERS.md             Extension checklist
│   ├── ZENODO_RELEASE.md       Upload steps
│   ├── preprint/manuscript.pdf
│   └── archive/commercial/     Historical outreach docs
└── tests/
```

---

## 5. Verified results

Cross-checked against `data/` on **2026-06-08**. Re-run [§8](#8-re-verify-locally) after new MEEP campaigns.

### Definitions

| Term | Definition |
|------|------------|
| **In-spec** | `\|split_ratio_upper − 0.5\| ≤ 0.05` at 1550 nm under `phase0_v1` |
| **Sim-budget B** | Exactly **B** MEEP simulations consumed by a policy |

### MEEP calibration

**Source:** `data/phase0/calibration_phase0_v1.json`

| Case | Split ratio |
|------|-------------|
| `empty_flip_y` / `full_flip_y` | **0.500** |
| `ref_published_flip_y` | **0.614** (expected in our template) |

### Champions

| ID | MEEP split | Source |
|----|------------|--------|
| `local_00022` | **0.500466** | `data/phase1/meep_search_local/top_candidates.csv` |
| `meep_bo_00128` | **0.509190** | `data/phase1/meep_search_deep/top_candidates.csv` |
| `meep_bo_00093` | **0.497288** | `data/phase0/sim_results_phase0_final.csv` |

### Sim-budget (six-seed pilot, B=100)

**Canonical:** [docs/SIM_BUDGET_REPLICATION_RESULTS.md](docs/SIM_BUDGET_REPLICATION_RESULTS.md)

| Policy | n_in_spec (mean ± std) |
|--------|-------------------------|
| **`surrogate_rank`** | **15.0 ± 3.2** |
| `hierarchical_35` | 14.8 ± 3.5 |
| `sigma_meep` | 10.6 ± 4.8 |

**Readout:** Supporting evidence for surrogate pre-filter at B=100; CIs overlap with `hierarchical_35`. Do not claim definitive ordering until n=20 replication completes.

### Release audits (negative results)

| Topic | Release file | v1 result |
|-------|--------------|-----------|
| Broadband flatness | `data/phase1/release/broadband_hunt.md` | 0 verified winners |
| IL / flux | `data/phase1/release/flux_il_audit.md` | Diagnostic, not product gate |
| Morph stress | `data/phase1/release/champion_fab_stress.md` | Sensitivity documented |

---

## 6. Implementation status

| Phase | Status | Notes |
|-------|--------|-------|
| **Phase 0** — prove loop | Complete | 512-label corpus, MEEP search |
| **Phase 1** — wedge + replication | Complete (six-seed) | n=20 optional extension |
| **Phase 2** — fab validation | Not started | See [docs/ROADMAP.md](docs/ROADMAP.md) |

Historical pilot/outreach collateral: [docs/archive/commercial/README.md](docs/archive/commercial/README.md) (not the public narrative).

---

## 7. Limitations

| Topic | Implication |
|-------|-------------|
| Surrogate R² negative on holdout | Pre-filter only; MEEP promotes |
| Heuristic DRC | Not foundry rule deck |
| Simulation-only | No fab correlation |
| Six-seed budget study | Label as pilot; n=20 in progress |
| Champions not all in 512-row corpus | Search finds winners outside batch labels |

---

## 8. Re-verify locally

```bash
source .venv/bin/activate

python -c "
import pandas as pd
df = pd.read_csv('data/phase0/sim_results_phase0_v1_all.csv')
print('rows', len(df), 'in_spec', ((df.split_ratio_upper-0.5).abs()<=0.05).sum())
"

python -c "import json; print('ranking_wins', json.load(open('data/phase1/wedge_a/ranking_eval.json'))['ranking_wins'])"

python scripts/check_preprint_v1_readiness.py
```

---

## 9. Command reference

### Primary workflows

| Goal | Command |
|------|---------|
| Smoke test | `python scripts/verify_setup.py` |
| Preprint + Zenodo bundle | `bash scripts/finalize_preprint_v1.sh` |
| Zenodo zip only | `bash scripts/build_zenodo_bundle.sh` |
| Wedge A sim-budget | `bash scripts/run_wedge_a.sh --full` |
| Aggregate replicates | `python scripts/aggregate_sim_budget_replicates.py --config configs/wedge_a_production.yaml` |
| MEEP batch | `bash scripts/run_meep.sh scripts/run_fdtd_batch.py --manifest …` |

Full script index: [scripts/README.md](scripts/README.md).

### Script groups

| Group | Examples |
|-------|----------|
| Decode / DRC | `decode_batch.py`, `verify_setup.py` |
| MEEP | `run_fdtd_batch.py`, `calibrate_meep.py` |
| Search | `meep_search.py`, `meep_search_local.py` |
| Surrogate | `train_wedge_a_surrogate.py`, `evaluate_surrogate_ranking.py` |
| Wedge A | `run_sim_budget_study.py`, `run_wedge_a_round.py` |
| Release | `build_repro_manifest.py`, `build_zenodo_bundle.sh`, `check_preprint_v1_readiness.py` |

---

## 10. Documentation index

| Document | Use when |
|----------|----------|
| [docs/OPEN_SOURCE_RELEASE.md](docs/OPEN_SOURCE_RELEASE.md) | Public framing, claim contract |
| [docs/ADOPTERS.md](docs/ADOPTERS.md) | Extending to new gates / topologies |
| [docs/ZENODO_RELEASE.md](docs/ZENODO_RELEASE.md) | Building and uploading Zenodo bundle |
| [docs/preprint/manuscript.pdf](docs/preprint/manuscript.pdf) | Citable methods and results |
| [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) | Architecture deep dive |
| [docs/SIM_BUDGET_REPLICATION_RESULTS.md](docs/SIM_BUDGET_REPLICATION_RESULTS.md) | Budget tables |
| [docs/sim_recipe_phase0.md](docs/sim_recipe_phase0.md) | MEEP recipe |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Issues, tests, PRs |

---

## 11. Contributing & release

- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md) — issues, `pytest tests/`, MEEP env notes.
- **Zenodo:** `bash scripts/build_zenodo_bundle.sh` → `data/phase1/release/nanophotonics_preprint_v1.zip` — steps in [docs/ZENODO_RELEASE.md](docs/ZENODO_RELEASE.md).
- **Citation:** [CITATION.cff](CITATION.cff).

---

*If a number in this README disagrees with a file under `data/`, trust the file and open an issue.*
