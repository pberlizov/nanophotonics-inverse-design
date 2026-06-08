#!/usr/bin/env python3
"""Quick r25/r50 comparison for mesh-stable recipe candidates on one mask."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nano_inv.crosscheck.meep_backend import run_meep
from nano_inv.crosscheck.types import SolverSpec

MASK = REPO / "data/phase1/meep_search_local/candidates/masks/local_00022_mask.npy"
REL = "data/phase1/meep_search_local/candidates/masks/local_00022_mask.npy"

CANDIDATES = [
    ("phase0_v1", "production"),
    ("phase0_v1_matgrid", "matgrid"),
    ("phase0_v1_refgrid100n", "refgrid100n+avg"),
    ("phase0_v1_refgrid25n", "refgrid25n"),
]


def main() -> None:
    mask = np.load(MASK)
    ref = None
    rows: list[tuple[str, float, float, float]] = []
    for recipe, label in CANDIDATES:
        r25 = run_meep("local_00022", REL, mask, SolverSpec(f"{label}_r25", "meep", recipe, 25, ""))
        r50 = run_meep("local_00022", REL, mask, SolverSpec(f"{label}_r50", "meep", recipe, 50, ""))
        if r25.status != "ok" or r50.status != "ok":
            print(f"{label}: ERROR r25={r25.status} r50={r50.status}")
            continue
        if ref is None and recipe == "phase0_v1":
            ref = r25.split_ratio_upper
        d_prod = abs(r25.split_ratio_upper - ref) if ref is not None else float("nan")
        gap = abs(r50.split_ratio_upper - r25.split_ratio_upper)
        rows.append((label, r25.split_ratio_upper, r50.split_ratio_upper, gap))
        print(
            f"{label:22s} r25={r25.split_ratio_upper:.4f} r50={r50.split_ratio_upper:.4f} "
            f"gap={gap:.4f} d_prod={d_prod:.4f}"
        )
    if rows:
        best = min(rows, key=lambda x: x[3])
        print(f"\nBest r25↔r50 gap: {best[0]} ({best[3]:.4f})")


if __name__ == "__main__":
    main()
