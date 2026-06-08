"""Phase 0 MEEP 2D TE template for mask → splitter metrics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    import meep as mp
except ImportError as exc:  # pragma: no cover
    mp = None  # type: ignore
    _MEEP_IMPORT_ERROR = exc
else:
    _MEEP_IMPORT_ERROR = None


@dataclass(frozen=True)
class MeepRecipe:
    wavelength_um: float = 1.55
    resolution: int = 25
    cell_size_um: float = 6.0
    design_size_um: float = 4.0
    pml_um: float = 0.5
    wg_width_um: float = 0.45
    eps_si: float = 12.0
    eps_sio2: float = 2.25
    arm_y_upper: float = 0.6
    arm_y_lower: float = -0.6
    arm_half_height_um: float = 0.25
    # Extend port Si blocks into the design box (µm); matches v1 junction overlap tuning.
    port_overlap_um: float = 0.0
    # Upsample mask pixels before MaterialGrid (e.g. 2 → 360×360 from 180×180).
    matgrid_upsample: int = 1
    # sdf / sdf_geom: smooth Heaviside edge width (µm); resolution-independent.
    sdf_smooth_um: float = 0.04
    decay_threshold: float = 1e-3
    decay_dt: int = 50
    # phase0_v1+
    mask_flip_y: bool = False
    min_runtime_factor: float = 40.0
    use_flux_decay: bool = True
    runtime_cap: int = 2000
    # Absolute Meep-time ceiling (all phases). Calibrated from v1 r25 (~2455).
    total_meep_time: float | None = None
    # phase0_v1_stable: minimum steps scale ~ (res/25)^2 before flux-decay stop
    min_runtime_resolution_scale: bool = False
    # nearest = legacy corpus contract; bilinear_fill = subpixel blend;
    # reference_grid = ε frozen on design_size×ref_res grid (mesh-stable cross-check).
    # material_grid = MEEP MaterialGrid from mask pixels (resolution-stable).
    # fullcell_matgrid = MaterialGrid raster of entire cell (ports + mask) at fixed px/µm.
    # sdf = signed-distance smooth ε in design region (continuous geometry).
    # sdf_geom = analytical port blocks + sdf design (Phase D0).
    mask_sampling: str = "nearest"
    mask_reference_res: int = 25
    # reference_grid only: bilinear vs nearest when rasterizing mask → ref ε
    refgrid_mask_bilinear: bool = False
    # reference_grid only: bilinear | nearest when sampling the frozen ε raster
    reference_interp: str = "bilinear"
    eps_averaging: bool = True
    # epsilon_file: rasterize ε at this px/µm into HDF5 once, then epsilon_input_file
    eps_file_res: int = 100

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MeepRecipe":
        known = cls.__dataclass_fields__
        return cls(**{k: d[k] for k in known if k in d})

    @classmethod
    def for_version(cls, version: str, base: dict[str, Any] | None = None) -> "MeepRecipe":
        """Recipe presets keyed by recipe_version string."""
        cfg = dict(base or {})
        if version == "phase0_v0":
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": False,
                    "min_runtime_factor": 20.0,
                    "decay_threshold": 1e-4,
                }
            )
        if version == "phase0_v1":
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                }
            )
        if version == "phase0_v2":
            # Template spike: tighter arm spacing + narrower input wg (calibrate before trusting).
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "wg_width_um": cfg.get("wg_width_um", 0.42),
                    "arm_y_upper": cfg.get("arm_y_upper", 0.55),
                    "arm_y_lower": cfg.get("arm_y_lower", -0.55),
                }
            )
        if version == "phase0_v1_stable":
            # Deprecated: flux-decay at one arm can run 10k+ steps without closing.
            return cls.for_version("phase0_v1_long", cfg)
        if version == "phase0_v1_long":
            # Fixed step budget ∝ res×(res/25) — no early flux-decay stop (mesh-safe).
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": False,
                    "min_runtime_factor": 80.0,
                    "min_runtime_resolution_scale": True,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "runtime_cap": 25000,
                }
            )
        if version == "phase0_v1_subpixel":
            # Bilinear mask fill + no MEEP edge averaging (experimental; r50 still diverged).
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "mask_sampling": "bilinear_fill",
                    "eps_averaging": False,
                }
            )
        if version == "phase0_v1_noavg":
            # Production nearest-pixel ε; disable MEEP edge averaging (mesh-stable).
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "eps_averaging": False,
                }
            )
        if version == "phase0_v1_refgrid":
            # ε sampled from fixed 25 px/µm design grid — r50 should match r25 contract.
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "mask_sampling": "reference_grid",
                    "mask_reference_res": int(cfg.get("mask_reference_res", 25)),
                    "eps_averaging": False,
                }
            )
        if version == "phase0_v1_matgrid":
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "mask_sampling": "material_grid",
                    "eps_averaging": False,
                }
            )
        if version == "phase0_v1_matgrid_avg":
            # MaterialGrid + MEEP subpixel averaging on grid edges (closer to production v1).
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "mask_sampling": "material_grid",
                    "eps_averaging": True,
                }
            )
        if version == "phase0_v1_sdf":
            # Continuous SDF geometry in design region; MEEP edge averaging on smooth ε.
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "mask_sampling": "sdf",
                    "sdf_smooth_um": float(cfg.get("sdf_smooth_um", 0.04)),
                    "eps_averaging": True,
                }
            )
        if version == "phase0_v1_sdf_noavg":
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "mask_sampling": "sdf",
                    "sdf_smooth_um": float(cfg.get("sdf_smooth_um", 0.04)),
                    "eps_averaging": False,
                }
            )
        if version == "phase0_v1_sdf_geom":
            # Analytical port blocks + SDF design (Phase D0 hybrid).
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "mask_sampling": "sdf_geom",
                    "sdf_smooth_um": float(cfg.get("sdf_smooth_um", 0.04)),
                    "eps_averaging": False,
                }
            )
        if version == "phase0_v1_matgrid_cal":
            # Matgrid + geometry from meep_bo_00128 calibration winner (arm=0.60, wg=0.43).
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "mask_sampling": "material_grid",
                    "eps_averaging": False,
                    "wg_width_um": float(cfg.get("wg_width_um", 0.43)),
                    "arm_y_upper": float(cfg.get("arm_y_upper", 0.60)),
                    "arm_y_lower": float(cfg.get("arm_y_lower", -0.60)),
                    "arm_half_height_um": float(cfg.get("arm_half_height_um", 0.24)),
                }
            )
        if version == "phase0_v1_sdf_cal":
            # SDF + calibrated port geometry (same knobs as matgrid_cal).
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "mask_sampling": "sdf",
                    "sdf_smooth_um": float(cfg.get("sdf_smooth_um", 0.04)),
                    "wg_width_um": float(cfg.get("wg_width_um", 0.43)),
                    "arm_y_upper": float(cfg.get("arm_y_upper", 0.60)),
                    "arm_y_lower": float(cfg.get("arm_y_lower", -0.60)),
                    "arm_half_height_um": float(cfg.get("arm_half_height_um", 0.24)),
                    "eps_averaging": True,
                }
            )
        if version == "phase0_v1_refgrid100n":
            # Fixed 100 px/µm ε raster; mesh-stable r25↔r50 (see tune_matgrid_recipe.py).
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "mask_sampling": "reference_grid",
                    "mask_reference_res": 100,
                    "refgrid_mask_bilinear": False,
                    "reference_interp": "bilinear",
                    "eps_averaging": True,
                }
            )
        if version == "phase0_v1_refgrid25n":
            # Coarser frozen grid (25 px/µm); closer to legacy refgrid r25≈r50 tests.
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "mask_sampling": "reference_grid",
                    "mask_reference_res": 25,
                    "refgrid_mask_bilinear": False,
                    "reference_interp": "nearest",
                    "eps_averaging": False,
                }
            )
        if version == "phase0_v1_epsfile":
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "mask_sampling": "epsilon_file",
                    "eps_file_res": int(cfg.get("eps_file_res", 100)),
                    "eps_averaging": False,
                }
            )
        if version == "phase0_v1_epsfile_fixed":
            # Frozen HDF5 ε + fixed Meep time (no flux-decay variance across resolutions).
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": False,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "mask_sampling": "epsilon_file",
                    "eps_file_res": int(cfg.get("eps_file_res", 100)),
                    "eps_averaging": False,
                    "total_meep_time": float(cfg.get("total_meep_time", 2520.0)),
                }
            )
        if version == "phase0_v1_fixed":
            # Production ε_func; same absolute Meep time @ r25 and r50 (B2 control).
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": False,
                    "min_runtime_factor": 40.0,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "total_meep_time": float(cfg.get("total_meep_time", 2520.0)),
                }
            )
        if version == "phase0_v1_fullcell_matgrid":
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "mask_sampling": "fullcell_matgrid",
                    "mask_reference_res": int(cfg.get("mask_reference_res", 100)),
                    "eps_averaging": False,
                }
            )
        if version == "phase0_v1_fullcell_matgrid_avg":
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "mask_sampling": "fullcell_matgrid",
                    "mask_reference_res": int(cfg.get("mask_reference_res", 100)),
                    "eps_averaging": True,
                }
            )
        if version == "phase0_v1_refgrid100":
            # Fixed ε at 100 px/µm (mask scale), bilinear mask → ref grid.
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "mask_sampling": "reference_grid",
                    "mask_reference_res": 100,
                    "refgrid_mask_bilinear": True,
                    "eps_averaging": False,
                }
            )
        if version == "phase0_v1_tcap":
            # Same ε as v1; cap total Meep time to r25 production (~2455) at every resolution.
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "total_meep_time": float(cfg.get("total_meep_time", 2520.0)),
                }
            )
        if version == "phase0_v1_tcap_subpixel":
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 40.0,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "mask_sampling": "bilinear_fill",
                    "eps_averaging": False,
                    "total_meep_time": float(cfg.get("total_meep_time", 2520.0)),
                }
            )
        if version == "phase0_v1_burnscale":
            # Same ε contract as v1; longer res-scaled burn-in before flux-decay (r50 mesh fix).
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": True,
                    "min_runtime_factor": 80.0,
                    "min_runtime_resolution_scale": True,
                    "decay_threshold": 1e-3,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                }
            )
        if version == "phase0_v1_subpixel_long":
            # Resolution-independent ε (bilinear) + fixed step budget (no early decay).
            return cls.from_dict(
                {
                    **cfg,
                    "use_flux_decay": False,
                    "min_runtime_factor": 100.0,
                    "min_runtime_resolution_scale": True,
                    "mask_flip_y": cfg.get("mask_flip_y", True),
                    "mask_sampling": "bilinear_fill",
                    "eps_averaging": False,
                    "runtime_cap": 30000,
                }
            )
        return cls.from_dict(cfg)


@dataclass
class MeepSimResult:
    status: str
    flux_in: float
    flux_out_upper: float
    flux_out_lower: float
    split_ratio_upper: float
    insertion_loss_db: float
    error: str = ""
    runtime_steps: int = 0


def require_meep() -> None:
    if mp is None:
        import sys

        exe = sys.executable
        hint = (
            "MEEP is not available in this Python interpreter.\n"
            f"  Current: {exe}\n"
            "  decode/manifold uses:  ~/nanophotonics-inverse-design/.venv  (no meep)\n"
            "  MEEP sims need:      conda env 'mp'\n\n"
            "Fix (one-time):\n"
            "  bash ~/nanophotonics-inverse-design/scripts/install_meep.sh\n"
            "  source ~/miniforge3/etc/profile.d/conda.sh\n"
            "  conda activate mp\n"
            "  python scripts/run_fdtd_batch.py ...\n\n"
            "Do NOT run run_fdtd_batch.py from .venv — see docs/MEEP_SETUP.md"
        )
        raise ImportError(hint) from _MEEP_IMPORT_ERROR


def _prepare_mask(mask: np.ndarray, recipe: MeepRecipe) -> np.ndarray:
    m = np.asarray(mask, dtype=np.float64)
    if recipe.mask_flip_y:
        m = np.flipud(m)
    return m


def _sample_mask(
    m: np.ndarray,
    p: Any,
    half: float,
    L: float,
) -> float:
    """Return Si fill fraction in [0, 1] at physical point p."""
    h, w = m.shape
    fx = (p.x + half) / L * w - 0.5
    fy = (p.y + half) / L * h - 0.5
    x0 = int(np.floor(fx))
    y0 = int(np.floor(fy))
    if x0 < -1 or y0 < -1 or x0 >= w or y0 >= h:
        return 0.0
    tx = fx - x0
    ty = fy - y0

    def at(ix: int, iy: int) -> float:
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return 0.0
        return 1.0 if m[iy, ix] > 0.5 else 0.0

    v00 = at(x0, y0)
    v10 = at(x0 + 1, y0)
    v01 = at(x0, y0 + 1)
    v11 = at(x0 + 1, y0 + 1)
    return (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11


def _sample_mask_nearest(m: np.ndarray, p: Any, half: float, L: float) -> float:
    h, w = m.shape
    ix = int((p.x + half) / L * w)
    iy = int((p.y + half) / L * h)
    ix = min(max(ix, 0), w - 1)
    iy = min(max(iy, 0), h - 1)
    return 1.0 if m[iy, ix] > 0.5 else 0.0


def _eps_from_fill(fill: float, recipe: MeepRecipe) -> float:
    return recipe.eps_sio2 + fill * (recipe.eps_si - recipe.eps_sio2)


def _build_mask_sdf_grid(m: np.ndarray) -> np.ndarray:
    """Signed distance (pixels): positive inside Si, negative in air."""
    si = m > 0.5
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(si) - distance_transform_edt(~si)


def _sample_grid_bilinear(grid: np.ndarray, p: Any, half: float, L: float) -> float:
    h, w = grid.shape
    fx = (p.x + half) / L * w - 0.5
    fy = (p.y + half) / L * h - 0.5
    x0 = int(np.floor(fx))
    y0 = int(np.floor(fy))
    tx = fx - x0
    ty = fy - y0

    def at(ix: int, iy: int) -> float:
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return 0.0
        return float(grid[iy, ix])

    v00 = at(x0, y0)
    v10 = at(x0 + 1, y0)
    v01 = at(x0, y0 + 1)
    v11 = at(x0 + 1, y0 + 1)
    return (1.0 - tx) * (1.0 - ty) * v00 + tx * (1.0 - ty) * v10 + (1.0 - tx) * ty * v01 + tx * ty * v11


def _fill_from_sdf_um(sdf_um: float, smooth_um: float) -> float:
    """Smooth Heaviside: 0 (air) → 1 (Si) across edge."""
    w = max(smooth_um, 1e-6)
    return float(0.5 * (1.0 + np.tanh(sdf_um / w)))


def _build_port_blocks(
    recipe: MeepRecipe,
    *,
    half: float,
    cell_half: float,
) -> list[Any]:
    """Analytical Si port blocks (input + output arms)."""
    si = mp.Medium(epsilon=recipe.eps_si)
    port_len = max(0.1, cell_half - half + recipe.port_overlap_um)
    arm_hw = recipe.arm_half_height_um
    zsize = 0.0
    return [
        mp.Block(
            material=si,
            center=mp.Vector3(-(half + port_len / 2.0), 0, 0),
            size=mp.Vector3(port_len, recipe.wg_width_um, zsize),
        ),
        mp.Block(
            material=si,
            center=mp.Vector3(half + port_len / 2.0, recipe.arm_y_upper, 0),
            size=mp.Vector3(port_len, 2.0 * arm_hw, zsize),
        ),
        mp.Block(
            material=si,
            center=mp.Vector3(half + port_len / 2.0, recipe.arm_y_lower, 0),
            size=mp.Vector3(port_len, 2.0 * arm_hw, zsize),
        ),
    ]


def _build_reference_eps_grid(
    m: np.ndarray,
    recipe: MeepRecipe,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize mask → ε on a fixed px/µm grid in the design square."""
    half = recipe.design_size_um / 2.0
    n = max(8, int(round(recipe.design_size_um * recipe.mask_reference_res)))
    x = np.linspace(-half, half, n)
    y = np.linspace(-half, half, n)
    h, w = m.shape
    L = recipe.design_size_um
    eps = np.full((n, n), recipe.eps_sio2, dtype=np.float64)
    for j, yv in enumerate(y):
        for i, xv in enumerate(x):
            p = type("P", (), {"x": xv, "y": yv})()
            fill = (
                _sample_mask(m, p, half, L)
                if recipe.refgrid_mask_bilinear
                else _sample_mask_nearest(m, p, half, L)
            )
            eps[j, i] = _eps_from_fill(fill, recipe)
    return x, y, eps


def _sample_reference_eps(
    ref_x: np.ndarray,
    ref_y: np.ndarray,
    ref_eps: np.ndarray,
    p: Any,
    *,
    interp: str = "bilinear",
) -> float:
    """Sample from the reference ε grid."""
    if p.x < ref_x[0] or p.x > ref_x[-1] or p.y < ref_y[0] or p.y > ref_y[-1]:
        return float(ref_eps[0, 0])
    fx = (p.x - ref_x[0]) / (ref_x[-1] - ref_x[0]) * (len(ref_x) - 1)
    fy = (p.y - ref_y[0]) / (ref_y[-1] - ref_y[0]) * (len(ref_y) - 1)
    if interp == "nearest":
        ix = int(np.clip(np.round(fx), 0, len(ref_x) - 1))
        iy = int(np.clip(np.round(fy), 0, len(ref_y) - 1))
        return float(ref_eps[iy, ix])
    x0 = int(np.floor(fx))
    y0 = int(np.floor(fy))
    tx = fx - x0
    ty = fy - y0
    x1 = min(x0 + 1, len(ref_x) - 1)
    y1 = min(y0 + 1, len(ref_y) - 1)
    e00 = ref_eps[y0, x0]
    e10 = ref_eps[y0, x1]
    e01 = ref_eps[y1, x0]
    e11 = ref_eps[y1, x1]
    return float(
        (1.0 - tx) * (1.0 - ty) * e00
        + tx * (1.0 - ty) * e10
        + (1.0 - tx) * ty * e01
        + tx * ty * e11
    )


def _eps_h5_cache_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "phase1" / "recipe_sensitivity" / "eps_h5"


def _eps_h5_paths(sample_key: str, recipe: MeepRecipe) -> Path:
    d = _eps_h5_cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{sample_key}_r{recipe.eps_file_res}.h5"


def _make_eps_at(
    m: np.ndarray,
    recipe: MeepRecipe,
    *,
    half: float,
    L: float,
    w: int,
    h: int,
    wg_hw: float,
    arm_hw: float,
    ref_grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    sdf_grid: np.ndarray | None = None,
    ports_in_eps: bool = True,
):
    """Build epsilon_func matching production / refgrid / sdf sampling."""
    px_um = L / float(w)

    def in_box(p: mp.Vector3, x0: float, x1: float, y0: float, y1: float) -> bool:
        return x0 <= p.x <= x1 and y0 <= p.y <= y1

    def eps_at(p: mp.Vector3) -> float:
        if ports_in_eps:
            if p.x < -half and abs(p.y) <= wg_hw:
                return recipe.eps_si
            if p.x > half:
                if abs(p.y - recipe.arm_y_upper) <= arm_hw:
                    return recipe.eps_si
                if abs(p.y - recipe.arm_y_lower) <= arm_hw:
                    return recipe.eps_si
        if in_box(p, -half, half, -half, half):
            if sdf_grid is not None:
                sdf_px = _sample_grid_bilinear(sdf_grid, p, half, L)
                fill = _fill_from_sdf_um(sdf_px * px_um, recipe.sdf_smooth_um)
                return _eps_from_fill(fill, recipe)
            if recipe.mask_sampling == "reference_grid" and ref_grid is not None:
                rx, ry, re = ref_grid
                return _sample_reference_eps(rx, ry, re, p, interp=recipe.reference_interp)
            if recipe.mask_sampling == "bilinear_fill":
                fill = _sample_mask(m, p, half, L)
                return _eps_from_fill(fill, recipe)
            ix = int((p.x + half) / L * w)
            iy = int((p.y + half) / L * h)
            ix = min(max(ix, 0), w - 1)
            iy = min(max(iy, 0), h - 1)
            return recipe.eps_si if m[iy, ix] > 0.5 else recipe.eps_sio2
        return recipe.eps_sio2

    return eps_at


def _ensure_frozen_epsilon_h5(
    m: np.ndarray,
    recipe: MeepRecipe,
    sample_key: str,
    *,
    freeze_recipe: MeepRecipe | None = None,
    verbose: bool = False,
) -> Path:
    """Write MEEP epsilon HDF5 at eps_file_res if missing (production ε logic)."""
    import os
    import shutil

    eps_file = _eps_h5_paths(sample_key, recipe)
    if eps_file.is_file():
        return eps_file

    fr = freeze_recipe or recipe
    L = fr.design_size_um
    half = L / 2.0
    wg_hw = fr.wg_width_um / 2.0
    arm_hw = fr.arm_half_height_um
    h, w = m.shape

    ref_grid = None
    if fr.mask_sampling == "reference_grid":
        ref_grid = _build_reference_eps_grid(m, fr)

    eps_at = _make_eps_at(
        m,
        fr,
        half=half,
        L=L,
        w=w,
        h=h,
        wg_hw=wg_hw,
        arm_hw=arm_hw,
        ref_grid=ref_grid,
    )

    cache_dir = _eps_h5_cache_dir()
    stem = f"{sample_key}_r{recipe.eps_file_res}"
    cwd = os.getcwd()
    try:
        os.chdir(cache_dir)
        cell = mp.Vector3(fr.cell_size_um, fr.cell_size_um, 0)
        sim = mp.Simulation(
            cell_size=cell,
            geometry=[],
            epsilon_func=eps_at,
            resolution=recipe.eps_file_res,
            default_material=mp.Medium(epsilon=fr.eps_sio2),
            dimensions=2,
            filename_prefix=stem,
            eps_averaging=fr.eps_averaging,
        )
        sim.init_sim()
        mp.output_epsilon(sim, frequency=0)
        matches = sorted(cache_dir.glob(f"{stem}-eps-*.h5"), key=lambda p: p.stat().st_mtime)
        if not matches:
            raise RuntimeError(f"MEEP did not write {stem}-eps-*.h5 in {cache_dir}")
        generated = matches[-1]
        if eps_file.is_file():
            eps_file.unlink()
        shutil.move(str(generated), str(eps_file))
    finally:
        os.chdir(cwd)
    if verbose:
        print(f"  froze ε → {eps_file} @ {recipe.eps_file_res} px/µm")
    return eps_file


def _build_fullcell_matgrid_weights(
    m: np.ndarray,
    recipe: MeepRecipe,
    *,
    half: float,
    L: float,
    w: int,
    h: int,
    wg_hw: float,
    arm_hw: float,
    cell_half: float,
) -> np.ndarray:
    """Binary Si weights for full cell at mask_reference_res px/µm (matches v1 topology)."""
    grid_res = max(8, int(recipe.mask_reference_res))
    n = max(8, int(round(recipe.cell_size_um * grid_res)))
    weights = np.zeros((n, n), dtype=np.float64)
    xs = np.linspace(-cell_half, cell_half, n)
    ys = np.linspace(-cell_half, cell_half, n)
    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            si = False
            if xv < -half and abs(yv) <= wg_hw:
                si = True
            elif xv > half:
                if abs(yv - recipe.arm_y_upper) <= arm_hw:
                    si = True
                elif abs(yv - recipe.arm_y_lower) <= arm_hw:
                    si = True
            elif -half <= xv <= half and -half <= yv <= half:
                ix = int((xv + half) / L * w)
                iy = int((yv + half) / L * h)
                ix = min(max(ix, 0), w - 1)
                iy = min(max(iy, 0), h - 1)
                si = bool(m[iy, ix] > 0.5)
            weights[j, i] = 1.0 if si else 0.0
    # MEEP MaterialGrid: axes (x, y) ↔ transpose from row=y, col=x indexing above.
    return weights.T


def _upsample_mask_nearest(m: np.ndarray, factor: int) -> np.ndarray:
    """Repeat pixels so MaterialGrid has finer design raster without changing topology."""
    f = max(1, int(factor))
    if f <= 1:
        return m
    return np.repeat(np.repeat(m, f, axis=0), f, axis=1)


def _build_geometry_material_grid(
    m: np.ndarray,
    recipe: MeepRecipe,
    *,
    half: float,
    cell_half: float,
    wg_hw: float,
    arm_hw: float,
) -> list[Any]:
    """Waveguides + design region as MEEP MaterialGrid (mesh-stable ε)."""
    si = mp.Medium(epsilon=recipe.eps_si)
    sio2 = mp.Medium(epsilon=recipe.eps_sio2)
    m = _upsample_mask_nearest(m, recipe.matgrid_upsample)
    h, w = m.shape
    L = recipe.design_size_um
    port_len = max(0.1, cell_half - half + recipe.port_overlap_um)
    # MEEP grid axes are (x, y); mask rows are y, cols are x.
    weights = np.asarray(m, dtype=np.float64).T
    design_grid = mp.MaterialGrid(
        mp.Vector3(w, h),
        sio2,
        si,
        weights=weights,
        do_averaging=recipe.eps_averaging,
    )
    zsize = 0.0
    port_blocks = [
        mp.Block(
            material=si,
            center=mp.Vector3(-(half + port_len / 2.0), 0, 0),
            size=mp.Vector3(port_len, recipe.wg_width_um, zsize),
        ),
        mp.Block(
            material=si,
            center=mp.Vector3(half + port_len / 2.0, recipe.arm_y_upper, 0),
            size=mp.Vector3(port_len, 2.0 * arm_hw, zsize),
        ),
        mp.Block(
            material=si,
            center=mp.Vector3(half + port_len / 2.0, recipe.arm_y_lower, 0),
            size=mp.Vector3(port_len, 2.0 * arm_hw, zsize),
        ),
    ]
    design_block = mp.Block(
        material=design_grid,
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(L, L, zsize),
    )
    # Ports after design so Si waveguides override mask at the junction (matches v1 ε_func).
    return [design_block, *port_blocks]


def _build_geometry_fullcell_matgrid(
    m: np.ndarray,
    recipe: MeepRecipe,
    *,
    half: float,
    cell_half: float,
    wg_hw: float,
    arm_hw: float,
) -> list[Any]:
    """Single MaterialGrid over the full cell — ports + design on one fixed raster."""
    h, w = m.shape
    L = recipe.design_size_um
    si = mp.Medium(epsilon=recipe.eps_si)
    sio2 = mp.Medium(epsilon=recipe.eps_sio2)
    weights = _build_fullcell_matgrid_weights(
        m,
        recipe,
        half=half,
        L=L,
        w=w,
        h=h,
        wg_hw=wg_hw,
        arm_hw=arm_hw,
        cell_half=cell_half,
    )
    nx, ny = weights.shape
    grid = mp.MaterialGrid(
        mp.Vector3(nx, ny),
        sio2,
        si,
        weights=weights,
        do_averaging=recipe.eps_averaging,
    )
    return [
        mp.Block(
            material=grid,
            center=mp.Vector3(0, 0, 0),
            size=mp.Vector3(recipe.cell_size_um, recipe.cell_size_um, 0),
        )
    ]


def simulate_mask(
    mask: np.ndarray,
    recipe: MeepRecipe,
    verbose: bool = False,
    *,
    sample_key: str = "mask",
    flux_y_scale: float | None = None,
    in_monitor_dx: float | None = None,
) -> MeepSimResult:
    """Run 2D TE FDTD for one binary mask (H×W), values in {0, 1}."""
    require_meep()

    m = _prepare_mask(mask, recipe)
    if m.ndim != 2:
        return MeepSimResult("error", np.nan, np.nan, np.nan, np.nan, np.nan, "mask_not_2d")
    h, w = m.shape
    if h != w:
        return MeepSimResult("error", np.nan, np.nan, np.nan, np.nan, np.nan, "mask_not_square")

    L = recipe.design_size_um
    half = L / 2.0
    cell_half = recipe.cell_size_um / 2.0
    wg_hw = recipe.wg_width_um / 2.0
    arm_hw = recipe.arm_half_height_um

    fcen = 1.0 / recipe.wavelength_um
    df = 0.1 * fcen

    use_epsfile = recipe.mask_sampling == "epsilon_file"
    use_matgrid = recipe.mask_sampling in ("material_grid", "fullcell_matgrid")
    use_sdf_geom = recipe.mask_sampling == "sdf_geom"
    use_sdf = recipe.mask_sampling == "sdf" or use_sdf_geom

    ref_grid: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if recipe.mask_sampling == "reference_grid":
        ref_grid = _build_reference_eps_grid(m, recipe)

    sdf_grid: np.ndarray | None = _build_mask_sdf_grid(m) if use_sdf else None

    geometry: list[Any] = []
    epsilon_func = None
    eps_input: str | None = None
    if use_epsfile:
        freeze = replace(recipe, mask_sampling="nearest", resolution=recipe.eps_file_res)
        eps_input = str(
            _ensure_frozen_epsilon_h5(m, recipe, sample_key, freeze_recipe=freeze, verbose=verbose).resolve()
        )
    elif use_matgrid:
        if recipe.mask_sampling == "fullcell_matgrid":
            geometry = _build_geometry_fullcell_matgrid(
                m,
                recipe,
                half=half,
                cell_half=cell_half,
                wg_hw=wg_hw,
                arm_hw=arm_hw,
            )
        else:
            geometry = _build_geometry_material_grid(
                m,
                recipe,
                half=half,
                cell_half=cell_half,
                wg_hw=wg_hw,
                arm_hw=arm_hw,
            )
    elif use_sdf_geom:
        geometry = _build_port_blocks(recipe, half=half, cell_half=cell_half)
        epsilon_func = _make_eps_at(
            m,
            recipe,
            half=half,
            L=L,
            w=w,
            h=h,
            wg_hw=wg_hw,
            arm_hw=arm_hw,
            ref_grid=ref_grid,
            sdf_grid=sdf_grid,
            ports_in_eps=False,
        )
    else:
        if use_sdf:
            epsilon_func = _make_eps_at(
                m,
                recipe,
                half=half,
                L=L,
                w=w,
                h=h,
                wg_hw=wg_hw,
                arm_hw=arm_hw,
                ref_grid=ref_grid,
                sdf_grid=sdf_grid,
                ports_in_eps=True,
            )
        else:
            epsilon_func = _make_eps_at(
                m,
                recipe,
                half=half,
                L=L,
                w=w,
                h=h,
                wg_hw=wg_hw,
                arm_hw=arm_hw,
                ref_grid=ref_grid,
            )

    cell = mp.Vector3(recipe.cell_size_um, recipe.cell_size_um, 0)
    pml_layers = [mp.PML(recipe.pml_um)]

    sources = [
        mp.Source(
            mp.GaussianSource(frequency=fcen, fwidth=df, is_integrated=True),
            component=mp.Ez,
            center=mp.Vector3(-cell_half + recipe.pml_um + 0.3, 0),
            size=mp.Vector3(0, recipe.wg_width_um, 0),
        )
    ]

    in_dx = 0.1 if in_monitor_dx is None else float(in_monitor_dx)
    fy = 0.8 if flux_y_scale is None else float(flux_y_scale)
    in_center = mp.Vector3(-half + in_dx, 0)
    out_u = mp.Vector3(half - 0.1, recipe.arm_y_upper)
    out_l = mp.Vector3(half - 0.1, recipe.arm_y_lower)
    fr = mp.Vector3(0, recipe.wg_width_um * fy, 0)

    sim_kw: dict[str, Any] = {
        "cell_size": cell,
        "boundary_layers": pml_layers,
        "geometry": geometry,
        "sources": sources,
        "resolution": recipe.resolution,
        "default_material": mp.Medium(epsilon=recipe.eps_sio2),
        "dimensions": 2,
    }
    if use_matgrid:
        sim_kw["eps_averaging"] = False
    elif use_sdf_geom:
        sim_kw["eps_averaging"] = False
    elif use_epsfile:
        sim_kw["epsilon_input_file"] = eps_input
        sim_kw["eps_averaging"] = False
    else:
        sim_kw["epsilon_func"] = epsilon_func
        sim_kw["eps_averaging"] = recipe.eps_averaging
    sim = mp.Simulation(**sim_kw)

    tran_in = sim.add_flux(
        fcen, 0, 1, mp.FluxRegion(center=in_center, size=fr, direction=mp.X)
    )
    tran_u = sim.add_flux(
        fcen, 0, 1, mp.FluxRegion(center=out_u, size=fr, direction=mp.X)
    )
    tran_l = sim.add_flux(
        fcen, 0, 1, mp.FluxRegion(center=out_l, size=fr, direction=mp.X)
    )

    min_runtime = max(100, int(recipe.min_runtime_factor * recipe.resolution))
    if recipe.min_runtime_resolution_scale:
        # Steps ∝ res × (res/25): same recipe family, finer mesh gets proportionally more steps.
        scale = max(1.0, recipe.resolution / 25.0)
        min_runtime = max(
            min_runtime,
            int(recipe.min_runtime_factor * recipe.resolution * scale),
        )
    t_cap = recipe.total_meep_time
    if t_cap is None:
        t_cap = float(recipe.runtime_cap)
    if verbose:
        print(
            f"  MEEP res={recipe.resolution} min_runtime={min_runtime} "
            f"decay={recipe.use_flux_decay} t_cap={t_cap}"
        )

    if recipe.use_flux_decay:
        decay_pt = mp.Vector3(out_u.x, out_u.y)
        # Burn-in before flux-decay; absolute until=t_cap (fixes r50 running to t≈4100+).
        sim.run(until_after_sources=min_runtime, until=t_cap)
        t_now = float(getattr(sim, "meep_time", lambda: 0)() or 0)
        if t_now < t_cap - 1.0:
            sim.run(
                until_after_sources=mp.stop_when_fields_decayed(
                    recipe.decay_dt, mp.Ez, decay_pt, recipe.decay_threshold
                ),
                until=t_cap,
            )
    else:
        sim.run(until_after_sources=min_runtime, until=t_cap)

    runtime_steps = int(getattr(sim, "meep_time", lambda: 0)() or 0)
    flux_in = mp.get_fluxes(tran_in)[0]
    flux_u = mp.get_fluxes(tran_u)[0]
    flux_l = mp.get_fluxes(tran_l)[0]
    p_out = max(flux_u, 0.0) + max(flux_l, 0.0)
    denom = flux_u + flux_l
    split = float(flux_u / denom) if denom > 1e-12 else float("nan")
    il_db = (
        float(-10 * np.log10(p_out / flux_in))
        if flux_in > 1e-12 and p_out > 0
        else float("nan")
    )

    return MeepSimResult(
        status="ok",
        flux_in=float(flux_in),
        flux_out_upper=float(flux_u),
        flux_out_lower=float(flux_l),
        split_ratio_upper=split,
        insertion_loss_db=il_db,
        runtime_steps=runtime_steps,
    )


@dataclass
class MeepBroadbandResult:
    status: str
    splits_by_wavelength: dict[float, float]
    insertion_loss_by_wavelength: dict[float, float]
    worst_split_error: float
    mean_insertion_loss_db: float
    error: str = ""


def simulate_mask_broadband(
    mask: np.ndarray,
    recipe: MeepRecipe,
    wavelengths_um: list[float],
    *,
    target_split: float = 0.5,
    verbose: bool = False,
) -> MeepBroadbandResult:
    """Run simulate_mask at each wavelength; return worst-case split error vs target."""
    from dataclasses import replace

    splits: dict[float, float] = {}
    ils: dict[float, float] = {}
    for wl in wavelengths_um:
        r = replace(recipe, wavelength_um=float(wl))
        res = simulate_mask(mask, r, verbose=verbose)
        if res.status != "ok" or not np.isfinite(res.split_ratio_upper):
            return MeepBroadbandResult(
                status="error",
                splits_by_wavelength={},
                insertion_loss_by_wavelength={},
                worst_split_error=float("nan"),
                mean_insertion_loss_db=float("nan"),
                error=res.error or "sim_failed",
            )
        splits[wl] = res.split_ratio_upper
        ils[wl] = res.insertion_loss_db

    worst = float(max(abs(s - target_split) for s in splits.values()))
    mean_il = float(np.nanmean(list(ils.values())))
    return MeepBroadbandResult(
        status="ok",
        splits_by_wavelength=splits,
        insertion_loss_by_wavelength=ils,
        worst_split_error=worst,
        mean_insertion_loss_db=mean_il,
    )
