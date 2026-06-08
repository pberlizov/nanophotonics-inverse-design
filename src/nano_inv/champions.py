"""Champion latent centers for on-manifold search."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nano_inv.latent import pad_latent_to_standard

DEFAULT_CHAMPION_LATENTS = [
    "data/phase0/latents/ref_published_latent.npy",
    "data/phase1/meep_search_local/candidates/latents/local_00022_latent.npy",
    "data/phase1/meep_search_deep/candidates/latents/meep_bo_00128_latent.npy",
]


def load_champion_centers(repo_root: Path, paths: list[str] | None = None) -> list[np.ndarray]:
    """Load padded standard latents for multi-center σ perturbation."""
    rels = paths or DEFAULT_CHAMPION_LATENTS
    out: list[np.ndarray] = []
    for rel in rels:
        p = repo_root / rel
        if p.is_file():
            out.append(pad_latent_to_standard(np.load(p).astype(np.float32)))
    if not out:
        ref = repo_root / "data/phase0/latents/ref_published_latent.npy"
        if ref.is_file():
            out.append(pad_latent_to_standard(np.load(ref).astype(np.float32)))
    return out
