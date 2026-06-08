"""Search config and loss helpers (no jax / drcgenerator / sklearn)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SearchMode = Literal["perturb", "perlin", "mixed"]


@dataclass(frozen=True)
class SearchConfig:
    target_split_ratio: float = 0.5
    tolerance: float = 0.05
    drc_penalty: float = 1.0
    require_drc: bool = True
    mode: SearchMode = "mixed"
    objective: str = "split"  # split | multi
    max_insertion_loss_db: float = 12.0
    weight_split: float = 1.0
    weight_il: float = 0.15


def split_ratio_loss(pred: float, target: float) -> float:
    return float(abs(pred - target))


def insertion_loss_penalty(il_db: float, *, max_il_db: float) -> float:
    if not (il_db == il_db):  # NaN
        return 1.0
    excess = max(0.0, float(il_db) - max_il_db)
    return excess / max(max_il_db, 1e-6)


def meep_search_loss(
    split_ratio_upper: float,
    insertion_loss_db: float,
    cfg: SearchConfig,
) -> float:
    loss = cfg.weight_split * split_ratio_loss(split_ratio_upper, cfg.target_split_ratio)
    if cfg.objective == "multi":
        loss += cfg.weight_il * insertion_loss_penalty(
            insertion_loss_db, max_il_db=cfg.max_insertion_loss_db
        )
    return float(loss)


def broadband_split_loss(
    splits_by_wavelength: dict[float, float],
    target: float,
    *,
    flatness_weight: float = 0.0,
) -> float:
    """Worst-case |split - target| across wavelength samples.

    Optional ``flatness_weight`` penalizes spread across λ (encourages flat R_up(λ),
    not only a good worst point with ripples elsewhere).
    """
    if not splits_by_wavelength:
        return 1.0
    errs = [abs(float(s) - target) for s in splits_by_wavelength.values()]
    worst = float(max(errs))
    if flatness_weight <= 0.0 or len(errs) < 2:
        return worst
    return worst + flatness_weight * float(np.std(errs))


def pareto_tuple(
    split_ratio_upper: float,
    insertion_loss_db: float,
    target_split: float,
    *,
    max_il_db: float,
) -> tuple[float, float, float]:
    """(split_err, il_penalty, combined_scalar) for logging / non-dominated sorting."""
    se = split_ratio_loss(split_ratio_upper, target_split)
    il = insertion_loss_penalty(insertion_loss_db, max_il_db=max_il_db)
    return se, il, se + 0.15 * il


def dominates(
    a: tuple[float, float],
    b: tuple[float, float],
) -> bool:
    """Minimize both objectives (split_err, il_penalty)."""
    return a[0] <= b[0] and a[1] <= b[1] and (a[0] < b[0] or a[1] < b[1])
