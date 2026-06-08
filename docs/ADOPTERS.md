# Next steps for adopters

This repository is a **starting point** for testing whether the MEEP-gated search pattern transfers to your gate, topology, or fabrication model. The preprint ([manuscript.pdf](preprint/manuscript.pdf)) documents what worked and what did not under one frozen contract.

## What we showed works (one gate)

Under `phase0_v1` at 1550 nm:

- **Pattern:** DRC-manifold sample → surrogate rank → MEEP verify → corpus
- **Positive result:** surrogate-ranked policies improve **verified 50/50 split yield per MEEP dollar** vs local/random baselines (six-seed pilot)
- **Structural finding:** in-spec designs exist both near-reference (low Hamming) and far-manifold (Perlin)

## What we did not solve

| Target | Status | Artifact |
|--------|--------|----------|
| C-band flatness | 0 verified winners | `data/phase1/release/broadband_hunt.md` |
| Morphology robustness | 0 pass @ ±10–20 nm stress | `data/phase1/release/morph_robust_hunt.json` |
| IL / transmission | T ~1–3%, IL 15–19 dB in-spec; monitors uncalibrated | `data/phase1/release/flux_il_audit.md` |
| Definitive sim-budget stats | Six-seed pilot only | `data/phase1/wedge_a/sim_budget_replication_stats.csv` |

Treat these as **boundary conditions**, not failures of the search loop itself.

## Adoption checklist

1. **Freeze the forward-model contract** — pin recipe ID, resolution, ports, polarization, monitors; version every label row.
2. **Define promotion gates explicitly** — separate narrowband split, C-band flatness, IL/T, morph stress; never promote on surrogate score alone.
3. **Refresh the surrogate on the corpus** — retrain rankers as verified rows accumulate; report Spearman on $|R_{\mathrm{up}}-0.5|$ and top-$k$ enrichment.
4. **Account MEEP budget per policy** — compare at matched $B$ with paired seed wins.
5. **Close the sim–fab gap deliberately** — for foundry-facing claims, add variability maps or fabrication-aware objectives; do not extrapolate uncorrected flux IL.

## Translation paths

| Your goal | Suggested upgrade |
|-----------|-------------------|
| Broadband WDM splitter | Multi-$\lambda$ objective + per-$\lambda$ verify; see Hansen et al. 2024 broadband splitter ID |
| Morph robustness | Stress in the inner loop (erosion/dilation or variation-aware TO) |
| Lower IL / higher T | Adjoint or multi-objective with calibrated monitors; split–IL penalty |
| New topology | New `drcgenerator` manifold or mask parameterization; keep MEEP as sign-off |
| Faster search | Replace mask MLP with neural-operator surrogate; keep MEEP shortlist verify |

## Minimal repro path

```bash
bash scripts/setup.sh && bash scripts/install_meep.sh
bash scripts/finalize_preprint_v1.sh          # CPU: figures + PDF + readiness
# Optional MEEP: bash scripts/run_meep.sh scripts/champion_fom_table.py
```

Release summaries: `data/phase1/release/*.md`  
Zenodo bundle: `data/phase1/release/nanophotonics_preprint_v1.zip`

## Get in touch

Open a GitHub issue with your gate definition and recipe pin — especially if you extend the pattern to a new device class.
