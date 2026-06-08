# Preprint v1 backlog

**Goal:** Complete **v1 engineering** (split-focused pipeline + release artifacts), then **IL multi-objective search**, then **one arXiv upload** — see [`PUBLICATION_PLAN.md`](PUBLICATION_PLAN.md).

**North star claim:** Under one MEEP contract on a DRC manifold, surrogate-ranked search improves verified split-ratio yield per sim dollar; **IL is then added as an explicit objective** before publication.

**Publication gate:** Do **not** upload until `loss_aware_hunt` Phase B completes (`check_publication_readiness.py`).

---

## Status legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Done — artifact in repo |
| 🔄 | Running — background terminal |
| 🔲 | Todo — blocks or polishes v1 |
| ⏭️ | Optional for v1 — nice but not blocking |

---

## Tier 0 — Scope & messaging (CPU)

| # | Item | Status | Owner / action |
|---|------|--------|----------------|
| 0.1 | Split-only promotion gate stated in abstract + limitations | ✅ | `manuscript.tex` |
| 0.2 | IL framed as diagnostic, not product metric | ✅ | §release checks + flux audit |
| 0.3 | Flux-monitor audit conclusion in paper | 🔲 | §transmission — run `finalize_preprint_v1.sh` |
| 0.4 | “Six-seed pilot; n=20 in progress” if replication incomplete | 🔲 | Abstract + budget table footnote |
| 0.5 | Phase 1 loss-aware roadmap (1 paragraph) | 🔲 | Limitations or conclusion |

---

## Tier A — Required artifacts (reviewer / credibility)

| # | Item | Status | Artifact | Refresh command |
|---|------|--------|----------|-----------------|
| A1 | Champion FOM (split, T, IL) | ✅ | `data/phase1/release/champion_fom_table.{md,json}` | `bash scripts/run_meep.sh scripts/champion_fom_table.py` |
| A2 | Flux / IL monitor audit | ✅ | `data/phase1/release/flux_il_audit.{md,json}` | `bash scripts/run_meep.sh scripts/flux_il_audit.py` |
| A3 | C-band baseline sweep (champions) | ✅ | `champion_broadband.{md,json}` | `bash scripts/run_meep.sh scripts/champion_broadband_sweep.py` |
| A4 | Mesh convergence (pixel vs SDF) | ✅ | `champion_mesh_convergence.{md,json}` + fig | `champion_mesh_convergence.py` / `--plot-only` |
| A5 | Sim-budget replication | 🔄 | `wedge_a/sim_budget/replicates/run_*` | Terminal 39 — `run_release_replication.sh` |
| A6 | Aggregated budget stats + paired wins | 🔲 | `sim_budget_replication_stats.csv`, `sim_budget_paired_wins.md` | After A5: `aggregate_sim_budget_replicates.py --config configs/wedge_a_release_replication.yaml` |
| A7 | Repro manifest | ✅ | `data/phase1/release/repro_manifest.json` | `python scripts/build_repro_manifest.py` |
| A8 | Broadband hunt (contribution) | 🔄 | `broadband_hunt/` + `release/broadband_hunt.md` | Terminal 43 — refine/explore running; 0 winners so far |

**v1 publish rule:** Tier A complete **or** explicitly label preliminary: six-seed budget pilot + “20-seed replication in progress.” Do **not** claim decisive policy ordering without A5–A6 at n≥20.

---

## Tier B — Strongly recommended (mostly done)

| # | Item | Status | Artifact |
|---|------|--------|----------|
| B1 | Fab stress (±10/20/30 nm) | ✅ | `champion_fab_stress.{md,json}` + fig |
| B2 | Surrogate validation regimes | ✅ | `surrogate_validation.{md,json}` |
| B3 | Proposal-pool ablation | ✅ | `ablation_proposal_pool.{md,json}` |
| B4 | Extended novelty + panels | ✅ | `novelty_extended.{md,json}`, `novelty_panels/` |
| B5 | Morph-robust search | 🔄 | `morph_robust_hunt/` | Terminal — `morph_robust_search.py` |
| B6 | R² deep work | ✅ | `wedge_a/r2_deep_work_log.md` |

---

## Tier C — Paper build (CPU)

| # | Item | Status | Command |
|---|------|--------|---------|
| C1 | Export preprint figures | ✅ | `python scripts/export_preprint_figures.py` |
| C2 | Copy MEEP figures → `docs/preprint/figures/` | ✅ | mesh, fab, broadband (in finalize script) |
| C3 | Build `manuscript.pdf` | ✅ | `cd docs/preprint && pdflatex manuscript.tex` ×2 |
| C4 | Readiness checker | 🔲 | `python scripts/check_preprint_v1_readiness.py` |
| C5 | One-shot finalize driver | 🔲 | `bash scripts/finalize_preprint_v1.sh` |

---

## Tier E — Phase B: IL objective (**blocks arXiv**)

| # | Item | Status | Command |
|---|------|--------|---------|
| E1 | Loss-aware hunt (multi objective) | 🔲 | `bash scripts/run_publication_pipeline.sh --launch-loss-aware` |
| E2 | Release report | 🔲 | `python scripts/loss_aware_report.py` |
| E3 | Tradeoff figure | 🔲 | `python scripts/export_loss_aware_figure.py` |
| E4 | Manuscript §loss-aware | 🔲 | After E1–E3 |
| E5 | `check_publication_readiness.py` green | 🔲 | Before upload |

## Tier D — Optional later

| Item | Notes |
|------|-------|
| n=20 + B=200 full table in main text | Update when A5 completes |
| Broadband winners in Fig. broadband right panel | When A8 finds passers |
| Morph-robust passers | Supplement if any `morph_pass` |
| 3D MEEP | Out of Phase 0 |

---

## Execution order (while terminals run)

### Now (CPU — no MEEP)

```bash
cd ~/nanophotonics-inverse-design && source .venv/bin/activate
bash scripts/finalize_preprint_v1.sh          # manifest, figures, PDF, readiness report
python scripts/check_preprint_v1_readiness.py # human-readable gate summary
```

### When replication terminal finishes (or hits run_07+)

```bash
python scripts/aggregate_sim_budget_replicates.py --config configs/wedge_a_release_replication.yaml
python scripts/export_preprint_figures.py   # set PRIMARY_REPLICATES in script if n>6
# Edit manuscript Table budget + abstract replicate count
bash scripts/finalize_preprint_v1.sh
```

### When broadband hunt finishes

```bash
python scripts/broadband_hunt.py --stage report
python scripts/export_broadband_contribution_figure.py
bash scripts/finalize_preprint_v1.sh
```

### When morph-robust finishes

```bash
# Results in data/phase1/morph_robust_hunt/morph_robust_summary.json
# Add 1–2 sentences to fab stress subsection if n_pass > 0
bash scripts/finalize_preprint_v1.sh
```

---

## v1 arXiv upload checklist

- [ ] `python scripts/check_preprint_v1_readiness.py` → all **blocking** items green or documented as preliminary
- [ ] `docs/preprint/manuscript.pdf` builds clean (14 pp target)
- [ ] All figures under `docs/preprint/figures/` referenced in tex exist
- [ ] Abstract does not claim low IL or C-band-flat champions
- [ ] Data availability points to `repro_manifest.json` + frozen configs
- [ ] Optional: zip `data/phase1/release/` summary JSONs for ancillary files

---

## Honest v1 framing (one paragraph for cover letter)

> We report simulation-only inverse design of silicon 1×2 splitters on a DRC manifold with a frozen MEEP gate. The contribution is methodology—surrogate-assisted ranking, replicated sim-budget comparison, and release checks—not foundry-validated low-loss devices. Champions achieve 50/50 at 1550 nm; C-band flatness and insertion loss remain open (documented with sweeps and a flux-monitor audit). Twenty-seed budget replication and broadband/morph hunts were in progress at submission.

---

## Related docs

- `RELEASE_CHECKLIST.md` — reviewer-request mapping (legacy)
- `BROADBAND_CONTRIBUTION.md` — broadband figure narrative
- `scripts/run_followup_priorities.sh` — four parallel MEEP workstreams
