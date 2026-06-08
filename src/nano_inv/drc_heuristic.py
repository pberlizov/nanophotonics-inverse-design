"""Phase 0 design sanity checks (not foundry DRC)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DRCHeuristicResult:
    passed: bool
    fill_ratio: float
    min_run_length: int
    reasons: list[str]


def _max_run_length_1d(line: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in line:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def check_mask_heuristic(
    mask: np.ndarray,
    *,
    min_fill: float = 0.08,
    max_fill: float = 0.92,
    min_run: int = 2,
) -> DRCHeuristicResult:
    """
    Fast plausibility checks on decoded binary mask (H, W).

    This is NOT a foundry rule deck — only filters obvious broken decodes.
    """
    reasons: list[str] = []
    m = np.asarray(mask)
    if m.ndim != 2:
        reasons.append(f"expected_2d_got_{m.ndim}d")
        return DRCHeuristicResult(False, 0.0, 0, reasons)

    unique = np.unique(m)
    if not np.all(np.isin(unique, [0, 1])):
        reasons.append("non_binary_values")

    fill = float(m.mean())
    if fill < min_fill:
        reasons.append(f"fill_too_low_{fill:.3f}")
    if fill > max_fill:
        reasons.append(f"fill_too_high_{fill:.3f}")

    row_runs = max((_max_run_length_1d(m[i]) for i in range(m.shape[0])), default=0)
    col_runs = max((_max_run_length_1d(m[j]) for j in range(m.shape[1])), default=0)
    min_run_len = min(row_runs, col_runs)
    if min_run_len < min_run:
        reasons.append(f"min_run_length_{min_run_len}_lt_{min_run}")

    passed = len(reasons) == 0
    return DRCHeuristicResult(passed, fill, min_run_len, reasons)
