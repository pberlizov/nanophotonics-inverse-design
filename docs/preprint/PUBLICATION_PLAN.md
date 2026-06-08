# Publication plan (Zenodo v1 + GitHub)

**Strategy:** Ship **v1.0-preprint** as an open research release — reproducible pipeline, preprint PDF, citation-stable Zenodo bundle — not a commercial wedge or arXiv-gated product launch.

---

## Release phases

```
Phase A — v1 engineering (done)          Phase B — optional extensions       Phase C — publish
────────────────────────────────         ─────────────────────────────       ─────────────────
Tier A release artifacts ✅              n=20 replication (in progress)      GitHub public
Six-seed sim-budget replication ✅       morph / IL hunts documented         Tag v1.0-preprint
Negative broadband / IL results ✅       new gates per ADOPTERS.md           Zenodo upload + DOI
finalize_preprint_v1.sh ✅               do not block v1 on n=20             Update CITATION.cff
```

**Do not block Zenodo on:** n=20 replication, broadband winners, morph robustness, or IL product gates — document limitations in the preprint instead.

---

## Phase A — v1 engineering (complete)

| Artifact | Path | Blocks Zenodo? |
|----------|------|----------------|
| Repro manifest | `data/phase1/release/repro_manifest.json` | Yes |
| Preprint PDF | `docs/preprint/manuscript.pdf` | Yes |
| Release audits | `data/phase1/release/*.md` | No (bundled) |
| Sim-budget stats | `docs/SIM_BUDGET_REPLICATION_RESULTS.md` | No |
| Readiness script | `scripts/check_preprint_v1_readiness.py` | Yes |

When refreshing: `bash scripts/finalize_preprint_v1.sh`

Checklist: `docs/preprint/V1_BACKLOG.md`, `docs/preprint/RELEASE_CHECKLIST.md`

---

## Phase B — extensions (adopter roadmap)

Follow [ADOPTERS.md](../ADOPTERS.md) and manuscript §9. Examples:

- **n=20 replication:** `N_REPLICATES=20 bash scripts/run_release_replication.sh` — update tables when done; do not claim definitive policy stats until then.
- **Broadband / IL / morph:** documented hunts under `data/phase1/release/` — negative results are publishable.
- **Foundry / 3D MEEP:** explicitly deferred.

---

## Phase C — Zenodo + GitHub release

1. `bash scripts/finalize_preprint_v1.sh`
2. `bash scripts/build_zenodo_bundle.sh`
3. `python scripts/check_preprint_v1_readiness.py` — must pass blocking gates
4. Make GitHub repo public; tag `v1.0-preprint`
5. Upload bundle per [ZENODO_RELEASE.md](../ZENODO_RELEASE.md)
6. Add Zenodo DOI to `CITATION.cff` and README badge

**Framing for v1 release:**

- Methods: sim-gated search, surrogate ranking, budget replication (six-seed pilot)
- Results: split gate success; documented failures on broadband, morph, IL
- Limits: simulation-only, not deployment-ready, not foundry-calibrated IL

---

## Commands (quick reference)

```bash
# Refresh artifacts + PDF + readiness
bash scripts/finalize_preprint_v1.sh

# Build Zenodo zip
bash scripts/build_zenodo_bundle.sh

# Status
python scripts/check_preprint_v1_readiness.py
```

See also: [OPEN_SOURCE_RELEASE.md](../OPEN_SOURCE_RELEASE.md), [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
