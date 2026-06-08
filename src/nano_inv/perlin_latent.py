"""JAX-free Perlin latent sampling for MEEP (mp) and .venv paths."""

from __future__ import annotations

import numpy as np
from noise import pnoise2

from drcgenerator import ebeam_latent_to_ambient_path, photolitho_latent_to_ambient_path

_REPEAT = 1024


def _interpolate_1d_numpy(x: np.ndarray, y: np.ndarray):
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    ones = np.ones_like(x)
    X = np.stack([x, ones], axis=1)
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    slope_a, intercept_b = theta[0], theta[1]

    def linear_model(x_val: float) -> float:
        return float(slope_a * x_val + intercept_b)

    return linear_model


def _latent_grid_size(
    x_microns: float,
    y_microns: float,
    *,
    model_flag: str,
) -> tuple[int, int]:
    if x_microns < 0 or y_microns < 0:
        raise ValueError(f"Micron values cannot be negative. Got x={x_microns}, y={y_microns}")

    if model_flag == "ebeam":
        path = ebeam_latent_to_ambient_path
        threshold = 3.0
    elif model_flag == "photolitho":
        path = photolitho_latent_to_ambient_path
        threshold = 2.0
    else:
        raise ValueError(f"Unknown model_flag: {model_flag}")

    mapping_data = np.load(path)
    latent_points = mapping_data[0]
    ambient_points = mapping_data[1]

    def to_ambient(um: float) -> float:
        return float(um * 1000.0 / 25.0)

    map_fn = _interpolate_1d_numpy(ambient_points, latent_points)
    lat_x_float = np.floor(map_fn(to_ambient(x_microns)) + 0.5)
    lat_y_float = np.floor(map_fn(to_ambient(y_microns)) + 0.5)
    lat_x = threshold if lat_x_float <= 0.0 else lat_x_float
    lat_y = threshold if lat_y_float <= 0.0 else lat_y_float
    return int(lat_x), int(lat_y)


def _generate_noise_map(
    size_x: int,
    size_y: int,
    scale: float,
    offset: tuple[float, float],
    dim: int,
) -> np.ndarray:
    noise_map = np.zeros((size_x, size_y), dtype=np.float32)
    for i in range(size_x):
        for j in range(size_y):
            x = i / scale + offset[0]
            y = j / scale + offset[1]
            if dim == 2:
                n = pnoise2(x, y, repeatx=_REPEAT, repeaty=_REPEAT, base=0)
            elif dim == 3:
                n = (
                    0.6 * pnoise2(x + 0.3, y, repeatx=_REPEAT, repeaty=_REPEAT, base=0)
                    + 0.4 * pnoise2(x, y + 0.5, repeatx=_REPEAT, repeaty=_REPEAT, base=1)
                )
            elif dim == 4:
                n = (
                    0.4 * pnoise2(x + 0.2, y + 0.1, repeatx=_REPEAT, repeaty=_REPEAT, base=0)
                    + 0.3 * pnoise2(2 * x, 2 * y, repeatx=_REPEAT, repeaty=_REPEAT, base=1)
                    + 0.3 * pnoise2(x * 0.5, y * 0.5, repeatx=_REPEAT, repeaty=_REPEAT, base=2)
                )
            else:
                raise ValueError(f"Unsupported dimensionality: {dim}. Expected 2, 3, or 4.")
            noise_map[i, j] = n

    n_min, n_max = noise_map.min(), noise_map.max()
    if n_max - n_min > 1e-8:
        noise_map = (noise_map - n_min) / (n_max - n_min)
    else:
        noise_map = np.zeros_like(noise_map)
    return noise_map[np.newaxis, ..., np.newaxis]


def sample_latent_perlin(
    *,
    x_microns: float = 4.0,
    y_microns: float = 4.0,
    scale: float = 3.5,
    offset: tuple[float, float] = (0.0, 0.0),
    dim: int = 3,
    model_flag: str = "ebeam",
) -> np.ndarray:
    """Perlin latent sample (matches drcgenerator notebook defaults, no JAX)."""
    size_x, size_y = _latent_grid_size(x_microns, y_microns, model_flag=model_flag)
    z = _generate_noise_map(size_x, size_y, scale, offset, dim)
    return np.asarray(z, dtype=np.float32)
