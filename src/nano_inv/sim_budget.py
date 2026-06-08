"""Sim-budget metrics and result helpers for wedge A."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class MeepRow:
    sample_id: str
    split_ratio_upper: float
    insertion_loss_db: float
    in_spec: bool
    sigma: float | None = None
    policy: str = ""
    meep_budget: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimBudgetSummary:
    policy: str
    meep_budget: int
    n_meep: int
    n_in_spec: int
    best_split: float
    best_sample_id: str
    best_abs_err: float
    mean_split: float
    p_hit_topk: float | None = None
    topk: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_in_spec(split: float, target: float, tol: float) -> bool:
    return bool(np.isfinite(split) and abs(split - target) <= tol)


def summarize_meep_rows(
    rows: list[MeepRow],
    *,
    policy: str,
    meep_budget: int,
    target: float,
    tol: float,
    topk: int | None = None,
) -> SimBudgetSummary:
    ok = [r for r in rows if np.isfinite(r.split_ratio_upper)]
    if not ok:
        return SimBudgetSummary(
            policy=policy,
            meep_budget=meep_budget,
            n_meep=0,
            n_in_spec=0,
            best_split=float("nan"),
            best_sample_id="",
            best_abs_err=float("nan"),
            mean_split=float("nan"),
        )
    splits = np.array([r.split_ratio_upper for r in ok])
    errs = np.abs(splits - target)
    best_i = int(np.argmin(errs))
    best = ok[best_i]
    n_in = sum(1 for r in ok if r.in_spec)

    p_hit = None
    if topk is not None and topk > 0:
        k = min(topk, len(ok))
        order = np.argsort(errs)[:k]
        p_hit = float(any(ok[i].in_spec for i in order))

    return SimBudgetSummary(
        policy=policy,
        meep_budget=meep_budget,
        n_meep=len(ok),
        n_in_spec=n_in,
        best_split=float(best.split_ratio_upper),
        best_sample_id=best.sample_id,
        best_abs_err=float(errs[best_i]),
        mean_split=float(np.mean(splits)),
        p_hit_topk=p_hit,
        topk=topk,
    )


def rows_to_dataframe(rows: list[MeepRow]) -> pd.DataFrame:
    return pd.DataFrame([r.to_dict() for r in rows])


def inverse_hit_rate_from_labels(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    target: float,
    tol: float,
    top_k: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Compare surrogate-ranked top-k vs random top-k on held-out MEEP labels."""
    n = len(y_true)
    k = min(top_k, n)
    err_pred = np.abs(scores - target) if scores.ndim == 1 else np.abs(scores)
    sur_idx = np.argpartition(err_pred, k - 1)[:k]
    rand_idx = rng.choice(n, size=k, replace=False)

    def _hit(idxs: np.ndarray) -> bool:
        return bool(np.any(np.abs(y_true[idxs] - target) <= tol))

    sur_hit = _hit(sur_idx)
    rand_hit = _hit(rand_idx)
    return {
        "top_k": k,
        "hit_surrogate_topk": sur_hit,
        "hit_random_topk": rand_hit,
        "ranking_wins": sur_hit and not rand_hit or (sur_hit == rand_hit and np.mean(np.abs(y_true[sur_idx] - target)) < np.mean(np.abs(y_true[rand_idx] - target))),
        "mean_abs_err_surrogate_topk": float(np.mean(np.abs(y_true[sur_idx] - target))),
        "mean_abs_err_random_topk": float(np.mean(np.abs(y_true[rand_idx] - target))),
        "n_in_spec_surrogate_topk": int(np.sum(np.abs(y_true[sur_idx] - target) <= tol)),
        "n_in_spec_random_topk": int(np.sum(np.abs(y_true[rand_idx] - target) <= tol)),
    }
