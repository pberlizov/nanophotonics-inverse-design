"""Wrapper around drcgenerator EBeam manifold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nano_inv.perlin_latent import sample_latent_perlin


@dataclass
class EBeamManifold:
    """Pretrained EBL generative manifold G(z) -> binary mask."""

    model: Any
    params: dict
    reference_latent: Any

    @classmethod
    def load(cls, rng_seed: int = 1903) -> "EBeamManifold":
        import jax
        import jax.numpy as jnp

        from drcgenerator import ebeam_ps_latent_space_ex_path
        from drcgenerator.models import EBeamModel

        model = EBeamModel()
        reference = jnp.load(ebeam_ps_latent_space_ex_path)
        rng = jax.random.PRNGKey(rng_seed)
        params = model.init(rng, reference)
        return cls(model=model, params=params, reference_latent=reference)

    def decode(self, latent: Any) -> Any:
        return self.model.apply(self.params, latent)

    def decode_soft(self, latent: Any) -> Any:
        """Continuous density before hard binarization (Path A grad)."""
        from drcgenerator.models import EBeamModel

        return self.model.apply(self.params, latent, method=EBeamModel.density)

    def decode_ste(self, latent: Any) -> Any:
        """Hard mask forward (``decode``); gradients through soft density."""
        import jax
        import jax.numpy as jnp

        soft = jnp.squeeze(self.decode_soft(latent))
        hard = jnp.squeeze(self.decode(latent)).astype(soft.dtype)
        return soft + jax.lax.stop_gradient(hard - soft)

    def decode_numpy(self, latent: np.ndarray) -> np.ndarray:
        import jax.numpy as jnp

        out = self.decode(jnp.asarray(latent))
        return np.asarray(jnp.squeeze(out))


def sample_latent_perturbation(
    reference: np.ndarray,
    rng: np.random.Generator,
    sigma: float = 0.08,
) -> np.ndarray:
    from nano_inv.latent import sample_latent_perturbation as _sample

    return _sample(reference, rng, sigma=sigma)
