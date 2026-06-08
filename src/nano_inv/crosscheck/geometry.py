"""Shared 2D permittivity grid matching ``meep_sim.MeepRecipe`` template."""

from __future__ import annotations

import numpy as np

from nano_inv.meep_sim import MeepRecipe, _prepare_mask, _sample_mask, _eps_from_fill


def build_eps_grid(
    mask: np.ndarray,
    recipe: MeepRecipe,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rasterize the MEEP epsilon_func on a uniform grid.

    Returns (x_coords, y_coords, z_coords, eps_3d) with shape (Nx, Ny, 1).
    """
    m = _prepare_mask(mask, recipe)
    h, w = m.shape
    L = recipe.design_size_um
    half = L / 2.0
    cell_half = recipe.cell_size_um / 2.0
    n = int(recipe.cell_size_um * recipe.resolution)
    x = np.linspace(-cell_half, cell_half, n)
    y = np.linspace(-cell_half, cell_half, n)
    z = np.array([0.0])

    eps = np.full((n, n), recipe.eps_sio2, dtype=np.float64)
    wg_hw = recipe.wg_width_um / 2.0
    arm_hw = recipe.arm_half_height_um

    X, Y = np.meshgrid(x, y, indexing="xy")
    in_wg = (X < -half) & (np.abs(Y) <= wg_hw)
    in_arm_u = (X > half) & (np.abs(Y - recipe.arm_y_upper) <= arm_hw)
    in_arm_l = (X > half) & (np.abs(Y - recipe.arm_y_lower) <= arm_hw)
    in_design = (X >= -half) & (X <= half) & (Y >= -half) & (Y <= half)

    eps[in_wg | in_arm_u | in_arm_l] = recipe.eps_si

    ix = np.clip(((X[in_design] + half) / L * w).astype(int), 0, w - 1)
    iy = np.clip(((Y[in_design] + half) / L * h).astype(int), 0, h - 1)
    if recipe.mask_sampling == "bilinear_fill":
        from types import SimpleNamespace

        fills = np.zeros(in_design.sum(), dtype=np.float64)
        pts = np.where(in_design)
        for k in range(len(fills)):
            p = SimpleNamespace(x=float(X[pts[0][k], pts[1][k]]), y=float(Y[pts[0][k], pts[1][k]]))
            fills[k] = _sample_mask(m, p, half, L)
        eps[in_design] = _eps_from_fill(fills, recipe)
    else:
        eps[in_design] = np.where(m[iy, ix] > 0.5, recipe.eps_si, recipe.eps_sio2)

    return x, y, z, eps[:, :, np.newaxis]
