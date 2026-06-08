# {{PILOT_TITLE}}

**On-manifold inverse design for silicon photonics** — layouts that hit spec where the published reference does not, including geometries outside the expert σ-tuning neighborhood.

---

## Problem

Experts tune around a **known-good template** (small latent perturbations). On our frozen MEEP recipe the published reference split is **~0.61**, not 50/50 — and random σ-perturbs rarely land in-spec (~3%). The interesting designs are **functionally better** and often **structurally different** from what hand-tuning would explore.

## Our approach

1. Search the full **DRC-feasible manifold** (`drcgenerator`) — not only a tight ball around one layout.  
2. **MEEP-native optimization** finds in-spec splits; ML surrogate **pre-filters** candidates before verification (optional).  
3. Characterize **distance from published reference** (Hamming / far-manifold Perlin designs).  
4. *(Supporting)* Sim-budget comparison at equal MEEP call counts once B=30/50/100 are complete.

**MEEP promotes designs. The surrogate does not sign off.**

## Benchmark result (internal)

| Target | 50/50 @ {{WAVELENGTH_NM}} nm (±{{SPLIT_TOLERANCE}}) |
| Recipe | `{{RECIPE_VERSION}}` |
| Ref published | ~0.61 split in our template (not in-spec) |
| Champions | ~0.50–0.51 split; σ-local winner **closer** to ref in pixels than typical perturb |
| Far-manifold | Perlin in-spec designs ~**50%** pixel Hamming from ref |
| Surrogate | Holdout **ranking** wins; **R² not** used as product metric |

*Attach `hamming_cdf.png` from novelty study + `sim_budget_curve.png` when full budgets are run.*

## Pilot offer ({{PILOT_DURATION_WEEKS}} weeks)

**You provide:** target spec, platform (EBL / photo), optional reference layout.  
**We deliver:** 5–10 MEEP-verified layouts, sim-budget report, masks + optional GDS, simulation contract (§what “qualified” means).

**Not included in base pilot:** foundry sign-off, fab correlation, full PDK port rebuild (available as follow-on).

## Ideal fit

- PIC startups & university groups with an upcoming **MPW** or paper deadline  
- Teams already using **MEEP / Lumerical / COMSOL** who want faster **passive component** iteration  
- **Not** a fit for bulk materials, chemistry, or non-EM inverse problems

---

**{{COMPANY_NAME}}** · {{CONTACT_EMAIL}}
