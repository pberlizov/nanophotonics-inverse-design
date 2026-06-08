"""Tidy3D backend (local ``Simulation.run``) using shared epsilon grid."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nano_inv.crosscheck.geometry import build_eps_grid
from nano_inv.crosscheck.types import CrosscheckResult, SolverSpec
from nano_inv.meep_sim import MeepRecipe


def _quasi_2d_z_extent_um(wavelength_um: float, min_steps_per_wvl: int) -> tuple[float, int]:
    """Z span large enough for Tidy3D mode solver (>=3 grid cells along z)."""
    n_ref = np.sqrt(12.0)
    dz_est = wavelength_um / (min_steps_per_wvl * n_ref)
    z_size = max(0.5, 4.0 * dz_est)
    nz = 5
    return z_size, nz


def run_tidy3d(
    sample_id: str,
    mask_path: str,
    mask: np.ndarray,
    spec: SolverSpec,
    *,
    reference_split: float | None = None,
    run_time_ps: float = 4.0,
    min_steps_per_wvl: int = 12,
    estimate_only: bool = False,
) -> CrosscheckResult:
    try:
        import tidy3d as td
    except ImportError as exc:
        return CrosscheckResult(
            sample_id=sample_id,
            mask_path=mask_path,
            solver=spec.name,
            status="skipped",
            split_ratio_upper=float("nan"),
            insertion_loss_db=float("nan"),
            flux_in=float("nan"),
            flux_out_upper=float("nan"),
            flux_out_lower=float("nan"),
            error=f"tidy3d not installed: {exc}",
        )

    recipe = MeepRecipe.for_version(
        spec.recipe_version,
        {"resolution": spec.resolution},
    )
    try:
        x, y, _z2d, eps2d = build_eps_grid(mask, recipe)
        half = recipe.design_size_um / 2.0
        wvl = recipe.wavelength_um
        fcen = td.C_0 / wvl
        z_size, nz = _quasi_2d_z_extent_um(wvl, min_steps_per_wvl)
        z_coords = np.linspace(-z_size / 2.0, z_size / 2.0, nz)
        eps3 = np.repeat(eps2d, nz, axis=2)

        medium = td.CustomMedium(
            permittivity=td.SpatialDataArray(
                eps3, coords={"x": x, "y": y, "z": z_coords}
            )
        )
        struct = td.Structure(
            geometry=td.Box(
                center=(0, 0, 0),
                size=(recipe.cell_size_um, recipe.cell_size_um, z_size),
            ),
            medium=medium,
        )
        src = td.ModeSource(
            center=(-half + 0.1, 0, 0),
            size=(0, recipe.wg_width_um, z_size),
            source_time=td.GaussianPulse(freq0=fcen, fwidth=0.1 * fcen),
            direction="+",
            mode_spec=td.ModeSpec(num_modes=1),
        )
        mons = [
            td.FluxMonitor(
                center=(-half + 0.15, 0, 0),
                size=(0, recipe.wg_width_um, z_size),
                freqs=[fcen],
                name="flux_in",
            ),
            td.FluxMonitor(
                center=(half - 0.1, recipe.arm_y_upper, 0),
                size=(0, recipe.wg_width_um, z_size),
                freqs=[fcen],
                name="flux_u",
            ),
            td.FluxMonitor(
                center=(half - 0.1, recipe.arm_y_lower, 0),
                size=(0, recipe.wg_width_um, z_size),
                freqs=[fcen],
                name="flux_l",
            ),
        ]
        sim = td.Simulation(
            size=(recipe.cell_size_um, recipe.cell_size_um, z_size),
            center=(0, 0, 0),
            grid_spec=td.GridSpec.auto(min_steps_per_wvl=min_steps_per_wvl, wavelength=wvl),
            structures=[struct],
            sources=[src],
            monitors=mons,
            run_time=run_time_ps * 1e-12,
            boundary_spec=td.BoundarySpec.all_sides(td.PML()),
        )
        import tidy3d.web as web

        try:
            web.test()
        except Exception as auth_exc:
            return CrosscheckResult(
                sample_id=sample_id,
                mask_path=mask_path,
                solver=spec.name,
                status="skipped",
                split_ratio_upper=float("nan"),
                insertion_loss_db=float("nan"),
                flux_in=float("nan"),
                flux_out_upper=float("nan"),
                flux_out_lower=float("nan"),
                error=(
                    "Tidy3D FDTD requires Flexcompute cloud API (tidy3d configure --apikey=…). "
                    f"Auth check failed: {auth_exc}"
                ),
            )

        out_hdf5 = Path("data/phase1/crosscheck") / f"{sample_id}_{spec.name}.hdf5"
        out_hdf5.parent.mkdir(parents=True, exist_ok=True)

        est_cost = None
        try:
            est_cost = float(web.estimate_cost(sim))
        except Exception:
            pass
        if estimate_only:
            return CrosscheckResult(
                sample_id=sample_id,
                mask_path=mask_path,
                solver=spec.name,
                status="estimate",
                split_ratio_upper=float("nan"),
                insertion_loss_db=float("nan"),
                flux_in=float("nan"),
                flux_out_upper=float("nan"),
                flux_out_lower=float("nan"),
                runtime_note=f"estimated_flex_credits={est_cost}",
            )

        sim_data = web.run(
            sim,
            task_name=f"crosscheck_{sample_id}",
            path=str(out_hdf5),
            verbose=False,
        )
        fi = float(abs(sim_data["flux_in"].flux.values.item()))
        fu = float(abs(sim_data["flux_u"].flux.values.item()))
        fl = float(abs(sim_data["flux_l"].flux.values.item()))
        denom = fu + fl
        split = float(fu / denom) if denom > 1e-15 else float("nan")
        il = float(-10 * np.log10((fu + fl) / fi)) if fi > 1e-15 and (fu + fl) > 0 else float("nan")
        err = None
        if reference_split is not None and np.isfinite(split):
            err = float(abs(split - reference_split))
        return CrosscheckResult(
            sample_id=sample_id,
            mask_path=mask_path,
            solver=spec.name,
            status="ok",
            split_ratio_upper=split,
            insertion_loss_db=il,
            flux_in=fi,
            flux_out_upper=fu,
            flux_out_lower=fl,
            reference_split=reference_split,
            abs_err_vs_reference=err,
            runtime_note=(
                f"tidy3d run_time={run_time_ps}ps res={spec.resolution} z_size_um={z_size:.3f}"
            ),
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
