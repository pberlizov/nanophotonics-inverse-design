#!/usr/bin/env python3
"""Quick check that Phase 0 environment and manifold decode work."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import jax
import jax.numpy as jnp

from drcgenerator import ebeam_ps_latent_space_ex_path
from nano_inv.manifold import EBeamManifold


def main() -> None:
    import drcgenerator

    print("drcgenerator:", drcgenerator.__file__)
    print("jax:", jax.__version__)

    m = EBeamManifold.load()
    z = jnp.load(ebeam_ps_latent_space_ex_path)
    mask = m.decode(z)
    print("reference latent:", tuple(z.shape))
    print("decoded mask:", tuple(mask.shape), "unique", jnp.unique(mask))

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from nano_inv.drc_heuristic import check_mask_heuristic
    import numpy as np

    r = check_mask_heuristic(np.asarray(jnp.squeeze(mask)))
    print("heuristic DRC:", r)


if __name__ == "__main__":
    main()
