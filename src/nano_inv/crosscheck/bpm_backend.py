"""Scalar 2D BPM reference solver (qualitative cross-check, not a product solver)."""

from __future__ import annotations

import numpy as np

from nano_inv.crosscheck.geometry import build_eps_grid
from nano_inv.crosscheck.types import CrosscheckResult, SolverSpec
from nano_inv.meep_sim import MeepRecipe


def run_bpm(
    sample_id: str,
    mask_path: str,
    mask: np.ndarray,
    spec: SolverSpec,
    *,
    reference_split: float | None = None,
    n_steps: int = 400,
) -> CrosscheckResult:
    """
    Paraxial BPM along +x on the shared epsilon grid.

    Uses |E|^2 in arm stripes at the output plane — useful for trend checks vs MEEP,
    not for absolute dB accuracy.
    """
    recipe = MeepRecipe.for_version(
        spec.recipe_version,
        {"resolution": spec.resolution},
    )
    try:
        x, y, z, eps3 = build_eps_grid(mask, recipe)
        eps = eps3[:, :, 0]
        n0 = np.sqrt(recipe.eps_sio2)
        k0 = 2 * np.pi / recipe.wavelength_um
        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])
        half = recipe.design_size_um / 2.0

        # Gaussian input in center waveguide
        wg_hw = recipe.wg_width_um / 2.0
        field = np.exp(-((y / (wg_hw * 0.85)) ** 2)).astype(np.complex128)
        field = field / np.sqrt(np.sum(np.abs(field) ** 2) * dy + 1e-30)

        for i in range(1, len(x)):
            n_eff = np.sqrt(np.maximum(eps[i, :], 1.0))
            phase = np.exp(1j * k0 * n_eff * dx)
            field = field * phase
            # Diffraction step
            ky = np.fft.fftfreq(len(y), d=dy) * 2 * np.pi
            spec_f = np.fft.fft(field)
            prop = np.exp(1j * (k0 * n0) ** 2 - ky**2).clip(max=0) * dx / (2 * n0 + 1e-30)
            prop = np.exp(1j * prop)
            field = np.fft.ifft(spec_f * prop)

        Y = y
        arm_hw = recipe.arm_half_height_um
        u_mask = np.abs(Y - recipe.arm_y_upper) <= arm_hw
        l_mask = np.abs(Y - recipe.arm_y_lower) <= arm_hw
        pu = float(np.sum(np.abs(field[u_mask]) ** 2) * dy)
        pl = float(np.sum(np.abs(field[l_mask]) ** 2) * dy)
        denom = pu + pl
        split = float(pu / denom) if denom > 1e-30 else float("nan")
        err = None
        if reference_split is not None and np.isfinite(split):
            err = float(abs(split - reference_split))
        return CrosscheckResult(
            sample_id=sample_id,
            mask_path=mask_path,
            solver=spec.name,
            status="ok",
            split_ratio_upper=split,
            insertion_loss_db=float("nan"),
            flux_in=float("nan"),
            flux_out_upper=pu,
            flux_out_lower=pl,
            reference_split=reference_split,
            abs_err_vs_reference=err,
            runtime_note=f"scalar_BPM steps={n_steps} (qualitative)",
        )
    except Exception as exc:
        return CrosscheckResult(
            sample_id=sample_id,
            mask_path=mask_path,
            solver=spec.name,
            status="error",
            split_ratio_upper=float("nan"),
            insertion_loss_db=float("nan"),
            flux_in=float("nan"),
            flux_out_upper=float("nan"),
            flux_out_lower=float("nan"),
            error=str(exc),
        )
