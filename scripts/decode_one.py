#!/usr/bin/env python3
"""Decode one latent to mask (requires .venv + drcgenerator)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.manifold import EBeamManifold  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent", type=Path, required=True)
    p.add_argument("--mask", type=Path, required=True)
    args = p.parse_args()

    m = EBeamManifold.load()
    z = np.load(args.latent)
    mask = m.decode_numpy(z)
    args.mask.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.mask, mask)


if __name__ == "__main__":
    main()
