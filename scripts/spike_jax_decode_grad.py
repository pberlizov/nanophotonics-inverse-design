#!/usr/bin/env python3
"""Phase D1 spike: JAX grad through drcgenerator decode_soft (no EM yet)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nano_inv.manifold import EBeamManifold

OUT = REPO / "data" / "phase1" / "jax_spike"


def main() -> None:
    manifold = EBeamManifold.load()
    z = manifold.reference_latent
    if z.ndim == 3:
        z = z[None, ...]

    def objective(latent: jnp.ndarray) -> jnp.ndarray:
        rho = manifold.decode_soft(latent)
        # Scalar proxy: mean soft density (any smooth functional proves ∂/∂z exists)
        return jnp.mean(rho)

    val, grad = jax.value_and_grad(objective)(z)
    gnorm = float(jnp.linalg.norm(grad))
    hard = manifold.decode(z)
    soft = manifold.decode_soft(z)

    result = {
        "objective_mean_rho": float(val),
        "grad_norm": gnorm,
        "grad_finite": bool(jnp.all(jnp.isfinite(grad))),
        "soft_shape": list(soft.shape),
        "hard_unique_values": [float(x) for x in jnp.unique(hard)],
        "mean_abs_hard_minus_soft": float(jnp.mean(jnp.abs(hard - soft))),
        "note": "Next: chain this with invrs-gym Ceviche loss (D2).",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "decode_soft_grad.json"
    out_path.write_text(json.dumps(result, indent=2))

    print("decode_soft grad spike")
    print(f"  mean(rho): {result['objective_mean_rho']:.4f}")
    print(f"  ||grad||: {result['grad_norm']:.4e}")
    print(f"  grad finite: {result['grad_finite']}")
    print(f"  mean|hard-soft|: {result['mean_abs_hard_minus_soft']:.4f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
