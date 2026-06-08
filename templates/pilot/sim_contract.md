# Simulation qualification contract

**Pilot:** {{PILOT_TITLE}}  
**ID:** `{{PILOT_ID}}`  
**Date:** {{CONTRACT_DATE}}  
**Provider:** {{COMPANY_NAME}}  
**Client:** {{CLIENT_NAME}}

---

## 1. Scope

This document defines what **“sim-qualified”** means for deliverables in this pilot. It does **not** constitute foundry DRC sign-off, guaranteed fabrication yield, or agreement with the Client’s internal electromagnetic solver until explicitly calibrated.

## 2. Device & platform

| Field | Value |
|-------|-------|
| Component | Power splitter (on-manifold inverse design) |
| Target split ratio | **{{TARGET_SPLIT}}** @ **{{WAVELENGTH_NM}}** nm |
| In-spec (sim) | {{IN_SPEC_DEFINITION}} |
| Platform | {{PLATFORM}} |
| Insertion loss (soft target) | ≤ **{{MAX_IL_DB}}** dB in search objective (not a hard fab guarantee) |

## 3. Forward model (MEEP)

| Parameter | Value |
|-----------|-------|
| Recipe version | `{{RECIPE_VERSION}}` |
| Resolution (cells/µm) | {{MEEP_RESOLUTION}} |
| Manifold | `drcgenerator` EBL 50/50 splitter family |
| DRC | Heuristic mask plausibility only — **not** a foundry rule deck |

All promoted designs are verified with **full-wave MEEP** under this recipe. Surrogate models are used **only** to rank candidates before verification.

## 4. Deliverables

1. **Ranked design dossier** — top layouts with mask arrays, PNG previews, optional GDS (layer 1/0), per-design `design_card.json`.
2. **Sim-budget report** — comparison at MEEP budgets **{{MEEP_BUDGETS}}** vs random and σ-only baselines at equal call count.
3. **This contract** — frozen for the pilot duration unless both parties agree in writing.

## 5. Exclusions (explicit)

- No warranty that fabricated devices meet electrical or optical spec.
- No liability for MPW shuttle scheduling, reticle fees, or process variation.
- Client-owned PDK integration, port calibration, and material parameters are **out of scope** unless added as a change order.
- Designs are searched **on-manifold** (feasible family); arbitrary freeform topology optimization is not offered in this pilot.

## 6. Acceptance

Client acceptance for pilot closeout: receipt of dossier + sim-budget report + readout call. Disputes on sim qualification are resolved by re-running MEEP on disputed `sample_id` under the recipe in §3.

---

**Contact:** {{CONTACT_EMAIL}}
