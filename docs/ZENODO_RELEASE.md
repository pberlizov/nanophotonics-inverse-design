# Zenodo release checklist (v1.0-preprint)

## What Zenodo is for

- **DOI** for the frozen preprint + figure bundle
- Citation-stable complement to the **living GitHub repo**

GitHub = code, issues, forks. Zenodo = snapshot you cite in papers.

## Build the bundle

```bash
bash scripts/build_zenodo_bundle.sh
```

Output: `data/phase1/release/nanophotonics_preprint_v1.zip`

Contents:

| File | Purpose |
|------|---------|
| `manuscript.pdf` | Preprint |
| `figures/*.pdf` | Publication figures |
| `champion_fom_table.md`, `flux_il_audit.md`, … | Release summaries |
| `CITATION.cff` | Citation metadata |
| `repro_manifest.json` | Artifact manifest |
| `LICENSE` | MIT |

## Upload steps

1. Log in at [zenodo.org](https://zenodo.org)
2. **New upload** → drag `nanophotonics_preprint_v1.zip` (or individual files)
3. **Upload type:** Publication → Preprint (or Software + Publication)
4. **Title:** MEEP-Gated Search on DRC Manifolds for Silicon 1×2 Power Splitters
5. **Authors:** from `CITATION.cff`
6. **Description:** paste abstract from `docs/preprint/manuscript.tex` (first ~200 words)
7. **License:** MIT (code) + CC-BY-4.0 for PDF if you prefer — match what you select on Zenodo
8. **Related identifiers:** link GitHub repo URL
9. **Keywords:** inverse design, MEEP, silicon photonics, surrogate model, DRC
10. **Version:** `v1.0-preprint`
11. Publish → copy DOI → update `CITATION.cff` and README Zenodo badge

## After upload

```bash
# Edit CITATION.cff: add doi: 10.5281/zenodo.XXXXX
# Edit README.md: replace DOI TBD with real DOI
bash scripts/build_zenodo_bundle.sh   # optional v1.0.1 bundle with DOI in CITATION.cff
git tag v1.0-preprint
git push origin v1.0-preprint
```

## What not to upload

- `.venv/`, raw `data/` MEEP dumps (too large; `data/` is gitignored)
- API keys, `~/.config/tidy3d/`
- Unfinished sprint logs unless you want them public on GitHub instead

Release MDs and manifest in the zip are the curated public data story.
