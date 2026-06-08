"""Latent sampling modes for MEEP-native search (no surrogate)."""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna

from nano_inv.latent import (
    LATENT_DIM,
    apply_latent_residual,
    flatten_latent,
    latent_from_pca_coeffs,
    pad_latent_to_standard,
    sample_latent_perturbation,
)

LatentSearchMode = str  # sigma | residual | pca | perlin


def load_perturb_latents_from_manifest(manifest_path: str, repo_root: Any) -> np.ndarray:
    import pandas as pd
    from pathlib import Path

    from nano_inv.manifest import filter_by_source

    m = pd.read_csv(Path(repo_root) / manifest_path)
    m = filter_by_source(m, "perturb")
    rows = []
    for p in m["latent_path"]:
        rows.append(np.load(Path(repo_root) / p))
    return np.stack(rows, axis=0)


def suggest_latent_for_meep(
    trial: optuna.Trial,
    reference: np.ndarray,
    rng: np.random.Generator,
    *,
    mode: LatentSearchMode = "residual",
    residual_dims: int = 12,
    residual_bound: float = 0.06,
    pca_bundle: tuple[np.ndarray, np.ndarray] | None = None,
    pca_bound: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """Return (latent, meta) for one Optuna trial."""
    ref = pad_latent_to_standard(reference)
    meta: dict = {"latent_mode": mode}

    if mode == "sigma":
        sigma = trial.suggest_float("sigma", 0.008, 0.04, log=True)
        z = sample_latent_perturbation(ref, rng, sigma=sigma)
        meta["sigma"] = sigma
        return pad_latent_to_standard(z), meta

    if mode == "residual":
        flat = flatten_latent(ref)
        deltas = np.zeros(LATENT_DIM, dtype=np.float32)
        n = min(residual_dims, LATENT_DIM)
        for i in range(n):
            deltas[i] = trial.suggest_float(f"dz_{i}", -residual_bound, residual_bound)
        z = apply_latent_residual(ref, deltas)
        meta["residual_dims"] = n
        meta["residual_bound"] = residual_bound
        return z, meta

    if mode == "pca":
        if pca_bundle is None:
            raise ValueError("pca mode requires pca_bundle=(mean_flat, components)")
        mean_flat, components = pca_bundle
        n_comp = components.shape[0]
        coeffs = np.array(
            [trial.suggest_float(f"pca_{i}", -pca_bound, pca_bound) for i in range(n_comp)],
            dtype=np.float32,
        )
        z = latent_from_pca_coeffs(mean_flat, components, coeffs)
        meta["pca_coeffs"] = coeffs.tolist()
        return z, meta

    if mode == "perlin":
        from nano_inv.perlin_latent import sample_latent_perlin

        scale = trial.suggest_float("perlin_scale", 2.5, 5.0)
        ox = trial.suggest_float("perlin_ox", -2.0, 2.0)
        oy = trial.suggest_float("perlin_oy", -2.0, 2.0)
        z = sample_latent_perlin(scale=scale, offset=(ox, oy), dim=3)
        meta["perlin_scale"] = scale
        meta["perlin_offset"] = (ox, oy)
        return pad_latent_to_standard(z), meta

    raise ValueError(f"unknown latent mode {mode!r}")
