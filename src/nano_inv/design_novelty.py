"""Geometric and functional distance vs published reference design."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nano_inv.latent import flatten_latent, pad_latent_to_standard
from nano_inv.surrogate import normalize_mask_to_standard

MASK_SHAPE = (180, 180)


@dataclass
class NoveltyMetrics:
    sample_id: str
    category: str
    mask_path: str
    hamming_fraction: float
    xor_pixel_count: int
    xor_fraction: float
    xor_largest_cc_fraction: float
    latent_l2: float | None
    split_ratio_upper: float | None
    in_spec: bool | None
    expert_sigma_ball: bool  # hamming <= typical expert σ-perturb exploration

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_reference_mask(ref_mask_path: Path) -> np.ndarray:
    return normalize_mask_to_standard(np.load(ref_mask_path))


def load_reference_latent_flat(ref_latent_path: Path) -> np.ndarray:
    return flatten_latent(np.load(ref_latent_path))


def xor_largest_component_fraction(xor: np.ndarray) -> float:
    """Fraction of XOR pixels in the largest 4-connected component."""
    try:
        from scipy import ndimage

        labeled, n = ndimage.label(xor)
        if n == 0:
            return 0.0
        sizes = ndimage.sum(xor, labeled, range(1, n + 1))
        return float(max(sizes) / max(xor.sum(), 1))
    except ImportError:
        return float(xor.mean())


def compute_novelty_metrics(
    *,
    sample_id: str,
    category: str,
    mask_path: Path,
    ref_mask: np.ndarray,
    ref_latent_flat: np.ndarray | None = None,
    latent_path: Path | None = None,
    split_ratio_upper: float | None = None,
    target: float = 0.5,
    tolerance: float = 0.05,
    expert_hamming_threshold: float = 0.06,
) -> NoveltyMetrics:
    mask = normalize_mask_to_standard(np.load(mask_path))
    ref = ref_mask.astype(bool)
    m = mask.astype(bool)
    xor = m ^ ref
    ham = float(xor.mean())
    xor_n = int(xor.sum())
    xor_cc = xor_largest_component_fraction(xor)

    latent_l2 = None
    if latent_path is not None and latent_path.exists() and ref_latent_flat is not None:
        z = flatten_latent(np.load(latent_path))
        latent_l2 = float(np.linalg.norm(z - ref_latent_flat))

    in_spec = None
    if split_ratio_upper is not None and np.isfinite(split_ratio_upper):
        in_spec = bool(abs(split_ratio_upper - target) <= tolerance)

    return NoveltyMetrics(
        sample_id=sample_id,
        category=category,
        mask_path=str(mask_path),
        hamming_fraction=ham,
        xor_pixel_count=xor_n,
        xor_fraction=ham,
        xor_largest_cc_fraction=xor_cc,
        latent_l2=latent_l2,
        split_ratio_upper=split_ratio_upper,
        in_spec=in_spec,
        expert_sigma_ball=ham <= expert_hamming_threshold,
    )
