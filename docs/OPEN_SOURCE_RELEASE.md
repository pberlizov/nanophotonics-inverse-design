# Open-source release

## Why this is open source

The value is the **reproducible search pattern** and **honest benchmark**, not a proprietary device. We are not pursuing commercialization of this wedge; the repo is meant for:

- Researchers testing MEEP-gated surrogate search on DRC-feasible manifolds
- Groups comparing sim-budget policies under a frozen forward model
- Adopters extending gates (broadband, morph, IL) with documented negatives as baselines

## Who this is for

| Audience | Use |
|----------|-----|
| Inverse-design researchers | Fork pipeline, swap gate/objective |
| Photonics grad students | Reproduce six-seed sim-budget pilot |
| ML-for-PDE folks | Ranking surrogate + FDTD verify loop |

Not aimed at: foundry sign-off, product insertion-loss specs, or turnkey PDK integration.

## Citation

```bibtex
@misc{berlizov2026meepgated,
  title  = {MEEP-Gated Search on DRC Manifolds for Silicon 1×2 Power Splitters},
  author = {Berlizov, Petr},
  year   = {2026},
  note   = {Preprint. Simulation-only study under frozen MEEP phase0\_v1 recipe.}
}
```

See [CITATION.cff](../CITATION.cff) for machine-readable metadata. Add Zenodo DOI after upload ([ZENODO_RELEASE.md](ZENODO_RELEASE.md)).

## GitHub + Zenodo

| | GitHub | Zenodo |
|---|--------|--------|
| Role | Living code, issues, PRs | Frozen citeable snapshot |
| Version | `main` + tags | `v1.0-preprint` DOI |
| Data | gitignored `data/`; scripts regenerate summaries | Bundle includes PDF + release MDs |

## Claim contract (please preserve)

**We claim:** verified split-ratio yield per sim dollar under `phase0_v1`; reproducible pipeline; documented negative results.

**We do not claim:** deployment-ready splitters, low IL, C-band WDM readiness, morph robustness, foundry-calibrated loss, definitive n=20 policy ordering.

## Historical commercial docs

[docs/WEDGE_A.md](WEDGE_A.md) and [docs/VALUE_PROPOSITION.md](VALUE_PROPOSITION.md) retain early wedge framing — **superseded** by this research release. See [docs/archive/commercial/README.md](archive/commercial/README.md).

## License

MIT — see [LICENSE](../LICENSE).
