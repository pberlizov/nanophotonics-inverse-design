"""Fabrication stress via morphology on an upsampled mask grid."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from nano_inv.surrogate import normalize_mask_to_standard

ISLAND_UM = 4.0
GRID = 180


def nm_per_px(*, upscale: int = 1) -> float:
    return ISLAND_UM * 1000.0 / GRID / max(1, upscale)


def disk_struct(radius_px: int) -> np.ndarray:
    if radius_px <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
    return (x * x + y * y) <= radius_px * radius_px


def upsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    m = normalize_mask_to_standard(mask).astype(bool)
    if factor <= 1:
        return m
    return np.kron(m, np.ones((factor, factor), dtype=bool))


def downsample_mask(fine: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return fine.astype(bool)
    h, w = fine.shape
    fh, fw = h // factor, w // factor
    fine = fine[: fh * factor, : fw * factor]
    blocks = fine.reshape(fh, factor, fw, factor)
    return blocks.mean(axis=(1, 3)) >= 0.5


def stress_radius_px(stress_nm: float, *, upscale: int) -> int:
    if stress_nm <= 0:
        return 0
    r = stress_nm / nm_per_px(upscale=upscale)
    return max(1, int(round(r)))


def apply_morph_stress(
    mask: np.ndarray,
    stress_nm: float,
    operation: str,
    *,
    upscale: int = 5,
) -> tuple[np.ndarray, int, float]:
    """Binary erode/dilate on upsampled grid, then vote-downsample to 180×180."""
    m = normalize_mask_to_standard(mask).astype(bool)
    fine = upsample_mask(m, upscale)
    r_px = stress_radius_px(stress_nm, upscale=upscale)
    se = disk_struct(r_px)
    if operation == "erode":
        stressed_fine = ndimage.binary_erosion(fine, structure=se)
    elif operation == "dilate":
        stressed_fine = ndimage.binary_dilation(fine, structure=se)
    else:
        raise ValueError(f"unknown operation {operation!r}")
    out = downsample_mask(stressed_fine, upscale)
    effective_nm = r_px * nm_per_px(upscale=upscale)
    return out.astype(float), r_px, effective_nm


def stress_variants(
    mask: np.ndarray,
    stress_nm_levels: tuple[int, ...] = (10, 20, 30),
    *,
    upscale: int = 5,
) -> list[tuple[str, np.ndarray, int, str, float]]:
    """(variant_id, mask, radius_px_fine, op, effective_nm)."""
    m = normalize_mask_to_standard(mask).astype(bool)
    out: list[tuple[str, np.ndarray, int, str, float]] = [
        ("nominal", m.astype(float), 0, "none", 0.0)
    ]
    for nm in stress_nm_levels:
        for op in ("erode", "dilate"):
            stressed, r_px, eff_nm = apply_morph_stress(m, float(nm), op, upscale=upscale)
            out.append((f"{op}_{nm}nm", stressed, r_px, op, eff_nm))
    return out
