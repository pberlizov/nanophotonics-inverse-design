#!/usr/bin/env python3
"""Fit PCA on perturb latents (.venv only); MEEP search loads the npz in mp."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.latent import fit_latent_pca  # noqa: E402
from nano_inv.meep_latent import load_perturb_latents_from_manifest  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/phase1_track_b.yaml")
    p.add_argument("--pca-dim", type=int, default=8)
    p.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data/phase1/track_b/latent_pca_basis.npz",
    )
    args = p.parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    latents = load_perturb_latents_from_manifest(cfg["data"]["manifest"], REPO_ROOT)
    mean_flat, components, evr = fit_latent_pca(latents, args.pca_dim)
    out = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, mean_flat=mean_flat, components=components, explained_variance_ratio=evr)
    print(f"wrote {out}  n_components={components.shape[0]}  evr_sum={evr.sum():.3f}")


if __name__ == "__main__":
    main()
