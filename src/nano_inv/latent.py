"""Latent shape helpers (no sklearn / joblib)."""

from __future__ import annotations

import numpy as np

LATENT_SHAPE = (1, 18, 18, 1)
LATENT_DIM = int(np.prod(LATENT_SHAPE))


def pad_latent_to_standard(latent: np.ndarray) -> np.ndarray:
    z = np.asarray(latent, dtype=np.float32)
    if z.shape == LATENT_SHAPE:
        return z
    if z.shape == (1, 17, 17, 1):
        out = np.zeros(LATENT_SHAPE, dtype=np.float32)
        out[0, :17, :17, 0] = z[0, :, :, 0]
        return out
    raise ValueError(f"unsupported latent shape {z.shape}")


def flatten_latent(latent: np.ndarray) -> np.ndarray:
    return np.ravel(pad_latent_to_standard(latent)).astype(np.float32)


def sample_latent_perturbation(
    reference: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 0.08,
) -> np.ndarray:
    """Gaussian perturbation around reference z (numpy only, no jax)."""
    eps = 1e-6
    noise = rng.normal(0.0, sigma, size=reference.shape).astype(np.float32)
    return np.clip(reference + noise, eps, 1.0)


def apply_latent_residual(
    reference: np.ndarray,
    deltas: np.ndarray,
    *,
    clip_eps: float = 1e-6,
) -> np.ndarray:
    """Add bounded residual on flattened standard latent, clip to (eps, 1)."""
    ref = pad_latent_to_standard(reference)
    flat = flatten_latent(ref)
    d = np.asarray(deltas, dtype=np.float32).ravel()
    if d.size != flat.size:
        raise ValueError(f"deltas size {d.size} != latent dim {flat.size}")
    out = np.clip(flat + d, clip_eps, 1.0).astype(np.float32)
    return out.reshape(LATENT_SHAPE)


def fit_latent_pca(
    latents: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA on flattened latents. Returns (mean_flat, components, explained_variance ratio).

    latents: (N, 1, 18, 18, 1) or (N, D)
    """
    from sklearn.decomposition import PCA

    if latents.ndim > 2:
        rows = [flatten_latent(z) for z in latents]
        X = np.stack(rows, axis=0)
    else:
        X = np.asarray(latents, dtype=np.float32)
    pca = PCA(n_components=min(n_components, X.shape[0], X.shape[1]))
    pca.fit(X)
    return (
        pca.mean_.astype(np.float32),
        pca.components_.astype(np.float32),
        pca.explained_variance_ratio_.astype(np.float32),
    )


def latent_from_pca_coeffs(
    mean_flat: np.ndarray,
    components: np.ndarray,
    coeffs: np.ndarray,
    *,
    clip_eps: float = 1e-6,
) -> np.ndarray:
    """z from PCA subspace: mean + coeffs @ components, clipped."""
    c = np.asarray(coeffs, dtype=np.float32).ravel()
    flat = np.clip(mean_flat + c @ components, clip_eps, 1.0).astype(np.float32)
    return flat.reshape(LATENT_SHAPE)
