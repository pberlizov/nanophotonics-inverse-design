# Value proposition (reframed)

> **Historical positioning — research release supersedes commercial framing.**  
> Public claim contract: [OPEN_SOURCE_RELEASE.md](OPEN_SOURCE_RELEASE.md). Do not use this doc for external sales pitches without matching honest limits.

**Last updated:** 2026-05-26  
**Verified metrics:** `data/phase1/novelty/novelty_report.json` — regenerate with `python scripts/characterize_design_novelty.py`

---

## Primary claim (lead with this)

> **We find on-manifold photonic layouts that meet optical spec in a frozen forward model where the published reference does not — including regimes a human expert would not explore by hand-tuning around that template.**

This is **not** primarily “fewer MEEP calls.” Sim-budget efficiency is **supporting evidence** once B=30/50/100 are complete.

---

## Three layers of evidence

### 1. Functional nonlinearity (σ-local champions)

| Design | Hamming vs `ref_published` | MEEP split @ `phase0_v1` | In-spec (0.5±0.05)? |
|--------|----------------------------|--------------------------|---------------------|
| `ref_published` | 0% | **0.614** | No |
| `local_00022` | **~0.47%** | **0.500** | Yes |
| `meep_bo_00128` | **~1.0%** | **0.509** | Yes |

**Interpretation:** A human expert perturbing σ around the published latent would explore masks that are **more** pixel-different from ref on average (~3.4% Hamming on perturb corpus) yet **rarely** land in-spec (3/100 perturb labels). The pipeline’s MEEP-native search finds **closer** micro-variants with **better** split — a non-obvious point on the mask→response map.

**Honest limit:** These champions are **not** “wildly different shapes” in pixel space. Do not pitch them as unrelated geometries.

### 2. Structural novelty (far-manifold / Perlin)

In-spec designs from **Perlin** sampling differ from ref by **~45–55%** Hamming (see `corpus_perlin_in_spec` in novelty CSV). That is **outside** the expert σ-perturbation ball (~≤6% Hamming threshold).

**Interpretation:** The same pipeline searches a **DRC-feasible manifold** that includes layouts no expert would reach by small edits to the published GDS-like template.

### 3. Search process (MEEP-native, not surrogate-first)

- **Promotion = MEEP.** Surrogate is an optional **pre-filter** to shortlist candidates before verification.
- **Production ranker** (`surrogate_improved`): **3,622** MEEP labels, σ feature; **ranking_wins** on merged corpus (top-20: **8** in-spec vs **3** random). **Holdout R² remains negative** — do not sell regression accuracy.
- Product story: **MEEP-driven search on-manifold** + ranker pre-filter; R² improvement is backlog (AL), not the pitch.

---

## Supporting claim (sim-budget @ B=100)

> At equal MEEP budget B=100, surrogate-ranked verification finds in-spec designs competitively with σ-perturbation and hierarchical expert search.

**Status:** **5× replication complete** — see [SIM_BUDGET_REPLICATION_RESULTS.md](SIM_BUDGET_REPLICATION_RESULTS.md) (`surrogate_rank` **15.0 ± 3.2** in-spec vs `hierarchical_35` **14.8 ± 3.5**; best \|split−0.5\| wins for surrogate). Validation replicate with `surrogate_improved`: `bash scripts/run_validation_replicate.sh`.

---

## What we do not claim

- Foundry DRC sign-off or fab yield
- Champions are pixel-distant from every human mental model of “splitter shape”
- Surrogate replaces MEEP
- Universal PIC inverse design without recipe calibration

---

## Outreach talk track (60 seconds)

1. Published ref + our MEEP template → wrong split; tiny on-manifold edits → 50/50. Experts don’t get there by σ-tuning alone.
2. We also search far from that template (Perlin family) where in-spec designs look nothing like the paper layout.
3. MEEP verifies everything; ML only ranks candidates.
4. Optional: sim-budget curve once 30/50/100 are run.

---

## Related artifacts

| File | Role |
|------|------|
| [novelty_summary.md](../data/phase1/novelty/novelty_summary.md) | Auto-generated tables |
| [WEDGE_A.md](WEDGE_A.md) | Technical wedge (updated) |
| [PILOT_README.md](PILOT_README.md) | Outreach playbook |
