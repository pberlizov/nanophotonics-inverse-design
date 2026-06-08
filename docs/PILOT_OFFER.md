# Pilot offer template

Copy into a proposal email or Doc. Replace bracketed fields.

---

## {{Company}} — Inverse design pilot proposal

**To:** [Client contact]  
**Re:** On-manifold inverse design pilot — [component name]  
**Duration:** 6 weeks (adjustable)  
**Fee:** [USD / EUR amount] fixed + pass-through compute

---

### Summary

[Company] will run a **6-week design sprint** for one passive photonic component on your target platform. We search inside a **DRC-feasible manifold**, rank candidates with a fast surrogate, and spend full-wave simulation (**MEEP**) only on verification. You receive a **ranked shortlist of layouts** qualified under a written **simulation contract**, plus a **sim-budget report** showing efficiency vs random and σ-only search at equal MEEP call counts.

### Scope (included)

| Item | Detail |
|------|--------|
| Kickoff | Target spec: split ratio, wavelength, tolerance, IL objective |
| Search | On-manifold perturbation + surrogate rank + MEEP verify |
| Deliverables | 5–10 designs: mask `.npy`, PNG, optional GDS, per-design metrics |
| Report | Sim-budget comparison at budgets [30, 50, 100] or as agreed |
| Contract | Frozen MEEP recipe + in-spec definition ([SIM_CONTRACT](templates/pilot/sim_contract.md)) |
| Readout | 60-minute technical review |

### Out of scope (base pilot)

- Foundry DRC sign-off and guaranteed fab yield
- Full PDK port rebuild or material calibration to your internal solver
- Active devices (modulators, lasers)
- Unlimited design iterations beyond agreed MEEP budget

### Client responsibilities

- Named technical point of contact
- Written target spec within 5 business days of kickoff
- Optional: reference layout or width rules (best-effort alignment)

### Timeline (indicative)

| Week | Milestone |
|------|-----------|
| 1 | Spec signed; sim contract agreed |
| 2 | Ranker trained on corpus + ranking gate |
| 3–4 | Sim-budget study + MEEP verification |
| 5 | Deliverable dossier + optional AL round |
| 6 | Readout + handoff |

### Pricing (fill in)

| Tier | Fee | Notes |
|------|-----|-------|
| Academic | $15k–$40k | Publication / MPW co-authorship negotiable |
| Startup | $50k–$150k | Fixed pilot; follow-on SOW for PDK/fab |
| Enterprise exploratory | $75k–$250k | Requires named component on roadmap |

Compute (AWS/local MEEP) billed at cost or included up to [N] core-hours.

### Acceptance

Pilot complete when deliverables in §Scope are received and readout held. Payment [50/50 milestone / net-30].

### Next step

Reply with your target spec and platform; we will return a filled `configs/pilot/<client>.yaml` and scheduled kickoff.

---

**Contact:** [email]
