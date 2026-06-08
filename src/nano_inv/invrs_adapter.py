"""Map drcgenerator EBL masks into invrs-gym Ceviche density parameters."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from totypes import types

from nano_inv.manifold import EBeamManifold
from drcgenerator.models import EBeamModel

ChallengeName = Literal[
    "ceviche_lightweight_power_splitter",
    "ceviche_power_splitter",
]


def get_ceviche_challenge(name: ChallengeName):
    import invrs_gym.challenges as gym_challenges

    if not hasattr(gym_challenges, name):
        raise ValueError(f"Unknown challenge {name!r}")
    return getattr(gym_challenges, name)()


def template_density(name: ChallengeName, seed: int = 0) -> types.Density2DArray:
    """Gym init params — carries fixed_void/fixed_solid/symmetries for the challenge."""
    challenge = get_ceviche_challenge(name)
    return challenge.component.init(jax.random.PRNGKey(seed))


def grid_size(name: ChallengeName) -> int:
    t = template_density(name)
    return int(t.array.shape[0])


def mask_to_density_array(
    mask: np.ndarray | None,
    *,
    grid_size: int,
    use_soft: bool = False,
    manifold: EBeamManifold | None = None,
    latent: np.ndarray | None = None,
) -> jnp.ndarray:
    """Resize 180×180 (4 µm) mask or soft density to Ceviche design grid."""
    if latent is not None:
        if manifold is None:
            manifold = EBeamManifold.load()
        z = jnp.asarray(latent)
        if z.ndim == 3:
            z = z[None, ...]
        field = manifold.decode_soft(z) if use_soft else manifold.decode(z)
        mask = np.asarray(jnp.squeeze(field))
    if mask is None:
        raise ValueError("mask required when latent is not provided")
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim > 2:
        m = np.squeeze(m)
    if m.max() > 1.5:
        m = m / float(m.max())
    m = np.clip(m, 0.0, 1.0)
    resized = jax.image.resize(
        jnp.asarray(m),
        shape=(grid_size, grid_size),
        method=jax.image.ResizeMethod.LINEAR,
    )
    return jnp.asarray(resized, dtype=jnp.float32)


def mask_to_gym_params(
    mask: np.ndarray | None = None,
    *,
    challenge: ChallengeName = "ceviche_lightweight_power_splitter",
    template_seed: int = 0,
    use_soft: bool = False,
    manifold: EBeamManifold | None = None,
    latent: np.ndarray | None = None,
) -> types.Density2DArray:
    """Build invrs-gym `Density2DArray` from a drcgenerator mask or latent."""
    if mask is None and latent is None:
        raise ValueError("Provide mask or latent")
    template = template_density(challenge, seed=template_seed)
    n = int(template.array.shape[0])
    if latent is not None:
        density = mask_to_density_array(
            None,
            grid_size=n,
            use_soft=use_soft,
            manifold=manifold,
            latent=latent,
        )
    else:
        density = mask_to_density_array(
            mask,
            grid_size=n,
            use_soft=False,
        )
    free = jnp.logical_not(jnp.logical_or(template.fixed_void, template.fixed_solid))
    merged = jnp.where(free, density, template.array)
    return replace(template, array=jnp.asarray(merged, dtype=jnp.float32))


def latent_to_gym_params(
    z: jnp.ndarray,
    template: types.Density2DArray,
    manifold: EBeamManifold,
) -> types.Density2DArray:
    """JAX-differentiable map z → gym Density2DArray (soft decode + resize)."""
    if z.ndim == 3:
        z = z[None, ...]
    z = jnp.clip(z, manifold.model.eps, 1.0)
    field = manifold.model.apply(manifold.params, z, method=EBeamModel.density)
    field2d = jnp.squeeze(field)
    n = int(template.array.shape[0])
    resized = jax.image.resize(
        field2d,
        shape=(n, n),
        method=jax.image.ResizeMethod.LINEAR,
    )
    free = jnp.logical_not(jnp.logical_or(template.fixed_void, template.fixed_solid))
    merged = jnp.where(free, resized, template.array)
    return replace(template, array=jnp.asarray(merged, dtype=jnp.float32))


def gym_loss_and_metric(
    z: jnp.ndarray,
    template: types.Density2DArray,
    manifold: EBeamManifold,
    challenge_name: ChallengeName,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Scalar loss and eval_metric for gradient-based refinement in z."""
    ch = get_ceviche_challenge(challenge_name)
    params = latent_to_gym_params(z, template, manifold)
    response, _aux = ch.component.response(params)
    loss = ch.loss(response)
    eval_metric = ch.eval_metric(response)
    return loss, eval_metric


def evaluate_gym_params(
    params: types.Density2DArray,
    *,
    challenge: ChallengeName = "ceviche_lightweight_power_splitter",
) -> dict[str, float]:
    """Run Ceviche FDFD eval (same API as gym optimization loop)."""
    ch = get_ceviche_challenge(challenge)
    response, aux = ch.component.response(params)
    loss = ch.loss(response)
    eval_metric = ch.eval_metric(response)
    metrics = ch.metrics(response, params, aux)
    return {
        "loss": float(loss),
        "eval_metric": float(eval_metric),
        "in_spec": float(eval_metric) >= 0.0,
        "binarization_degree": float(metrics.get("binarization_degree", float("nan"))),
    }
