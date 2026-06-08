# Broadband contribution (preprint section draft)

## Claim (target)

We show that **single-wavelength inverse design** can hit 50/50 at 1550 nm under a frozen MEEP template but is **strongly dispersive** across the C-band (worst |R_up−0.5| ≈ 0.36–0.47 on promoted champions). A **broadband-aware latent refinement** stage (worst-λ + flatness objective) can discover masks with **flat R_up(λ)** over 1530–1570 nm while staying on the DRC manifold.

## Methods (one paragraph)

After narrowband promotion (`|R_up(1.55 µm)−0.5| ≤ 0.05`), we re-score candidates on a uniform grid λ ∈ [1.53, 1.57] µm (step 5 nm for search, 5 nm for verification). The broadband loss is L = max_λ |R_up(λ)−0.5| + w·std_λ(R_up), with w=0.35, plus a soft IL prior. Optuna TPE refines 16 residual latent dimensions around each champion center (25 trials × 4 centers) and an optional 40-trial residual exploration. Release gate: worst |R_up−0.5| ≤ 0.05 over the band.

## Figures / tables

| Artifact | Path |
|----------|------|
| Narrowband R_up(λ) | `data/phase1/release/champion_broadband.png` |
| Before/after panel | `docs/preprint/figures/broadband_contribution.png` |
| Hunt winners | `data/phase1/release/broadband_hunt.md` |

## Suggested LaTeX subsection

```latex
\subsection{C-band flatness and broadband-aware refinement}
\label{sec:broadband}

Single-frequency inverse design achieves $|R_{\mathrm{up}}(1.55\,\mu\mathrm{m})-0.5|\le 0.05$ for all promoted champions (Table~\ref{tab:champions}), but wavelength sweeps reveal strong dispersion: worst-case split error over 1530--1570\,nm reaches $0.36$--$0.47$ (Fig.~\ref{fig:broadband}).
We therefore add a broadband refinement stage that minimizes $\max_\lambda |R_{\mathrm{up}}(\lambda)-0.5|$ plus a flatness penalty on $\mathrm{std}_\lambda R_{\mathrm{up}}$.
[Fill in: $N$ verified broadband-flat designs; best worst-case error; comparison to $\sigma$-only local search.]
```

## Run (MEEP)

```bash
# Full hunt (~2–6 h)
bash scripts/run_broadband_hunt.sh

# Pilot (~1 h)
TRIALS_PER_CENTER=10 SKIP_EXPLORE=1 bash scripts/run_broadband_hunt.sh

# Re-export figure after hunt
python scripts/export_broadband_contribution_figure.py
```

## If hunt finds zero winners

Report honestly as a **negative result + method contribution**:

1. Narrowband ID is insufficient for C-band specs under this template.
2. Broadband objective and verification pipeline are reproducible (`configs/broadband_hunt.yaml`).
3. Report **best achievable** worst-λ error after refinement (even if > 0.05) as a lower bound.

This is still publishable with the contrast figure (left panel only + hunt best-so-far dashed).
