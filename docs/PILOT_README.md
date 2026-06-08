# Pilot playbook — outreach-ready package

**Purpose:** Everything needed to **reach out** for a paid design sprint without improvising scope, deliverables, or sim language.

**Quick start**

```bash
cd ~/nanophotonics-inverse-design
source .venv/bin/activate

# Build collateral from existing wedge_a results (~2 min, no MEEP)
bash scripts/run_pilot.sh --config configs/pilot/benchmark_50_50.yaml --report-only

# Before external claims: full sim-budget (30/50/100 MEEP, hours)
bash scripts/run_pilot.sh --config configs/pilot/benchmark_50_50.yaml --full-meep --skip-train
```

**Output locations**

| Path | Contents |
|------|----------|
| `data/pilot/benchmark_50_50/deliverables/` | Dossier, SIM_CONTRACT, report, plot |
| `data/pilot/benchmark_50_50/outreach/` | Email-ready bundle + ONE_PAGER |

---

## Who to contact (ICP)

| Tier | Who | Why | First ask |
|------|-----|-----|-----------|
| 1 | University / national-lab **PIC** group | Low procurement; accepts sim-qualified | “Upcoming MPW or paper splitter?” |
| 1 | **PIC startup** (10–80 ppl) | Needs passive component throughput | “How many FDTD runs per acceptable coupler?” |
| 2 | Photonics **R&D** inside larger OEM | Budget for exploratory PoC | Named component on roadmap |
| 3 | MPW coordinator / cloud FDTD platform | Partnership, not first check | After one customer reference |

**Do not lead with “materials science.”** Only take metasurface / flat-optics meetings if they already do **EM + lithography**. Bulk chemistry / DFT is a different product.

---

## What you sell

**Product:** On-manifold inverse design pilot — MEEP-verified layouts that meet spec where your reference does not, including **far-from-template** regions of the manifold (see [VALUE_PROPOSITION.md](VALUE_PROPOSITION.md)).

**Supporting:** Sim-budget report at equal MEEP calls — **only after** `bash scripts/run_full_sim_budget.sh`.

**Not:** surrogate-only sign-off, “wildly different” pixel claim for σ-local champions, fab guarantee, full PDK integration (unless change order).

See [PILOT_OFFER.md](PILOT_OFFER.md) for scope & pricing placeholders, [PILOT_SOW.md](PILOT_SOW.md) for IP/data terms.

---

## Outreach checklist

See also [PILOT_OUTREACH_CHECKLIST.md](PILOT_OUTREACH_CHECKLIST.md).

Before the first email:

- [ ] Fill `contact_email` and `company_name` in `configs/pilot/benchmark_50_50.yaml`
- [ ] Run `bash scripts/run_pilot.sh --report-only` and open `outreach/ONE_PAGER.md`
- [ ] Run `--full-meep` if your slide only shows B=15,30 (pilot budgets)
- [ ] Attach: ONE_PAGER, `sim_budget_curve.png`, 1–2 champion PNGs from `deliverables/designs/`
- [ ] Do **not** claim fab correlation or PDK match until calibrated

First call agenda (30 min):

1. Their spec (split, λ, bandwidth, IL, footprint)
2. Platform (EBL vs photo, rule deck if any)
3. Their baseline (MEEP runs per acceptable design today)
4. Pilot timeline & exclusions (SIM_CONTRACT)

---

## Pipeline map

```mermaid
flowchart LR
  cfg[configs/pilot/*.yaml]
  train[train_wedge_a_surrogate]
  meep[run_sim_budget_study]
  dlv[build_pilot_deliverables]
  rpt[generate_pilot_report]
  pkg[package_pilot_outreach]

  cfg --> train --> meep --> dlv --> rpt --> pkg
```

| Step | Script | Notes |
|------|--------|-------|
| Config | `configs/pilot/benchmark_50_50.yaml` | Copy `client_template.yaml` per customer |
| Train | `train_wedge_a_surrogate.py` | Perturb `latent_mlp` ranker |
| MEEP | `run_sim_budget_study.py` | `--pilot` or full budgets |
| Dossier | `build_pilot_deliverables.py` | Champions + sim-budget top-k |
| Report | `generate_pilot_report.py` | Qualified claims language |
| Bundle | `package_pilot_outreach.py` | `outreach/` folder |

---

## Customizing for a client

1. Copy `configs/pilot/client_template.yaml` → `configs/pilot/acme.yaml`
2. Edit `pilot.client_name`, `targets`, `deliverables.top_k`
3. After kickoff: `bash scripts/run_pilot.sh --config configs/pilot/acme.yaml`
4. Point `sources.metrics_path` at client-specific outputs when isolated

Optional follow-ons (priced separately):

- Client MEEP/Lumerical recipe calibration
- Second component (e.g. 70/30 — see `configs/phase2_splitter_70_30.yaml`)
- MPW correlation round (fab not in base pilot)

---

## Related docs

| Doc | Role |
|-----|------|
| [WEDGE_A.md](WEDGE_A.md) | Technical wedge & metrics |
| [PILOT_OFFER.md](PILOT_OFFER.md) | Proposal template |
| [PILOT_SOW.md](PILOT_SOW.md) | IP & data |
| [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | North star |
| [configs/pilot/README.md](../configs/pilot/README.md) | Config reference |
