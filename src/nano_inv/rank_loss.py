"""Pairwise ranking losses for surrogate training (RankNet / LambdaRank-style weights)."""

from __future__ import annotations

import numpy as np


def rank_weighted_regression_weights(
    split_ratio_upper: np.ndarray,
    *,
    target_split_ratio: float = 0.5,
    power: float = 1.0,
) -> np.ndarray:
    """
    Emphasize near-target rows without torch: weight ∝ (1 / (ε + |err|))^power.
    """
    err = np.abs(np.asarray(split_ratio_upper, dtype=float) - target_split_ratio)
    return (1.0 / (0.02 + err)) ** power


def pairwise_rank_loss_torch(
    pred: "object",
    y: "object",
    *,
    target_split_ratio: float = 0.5,
    mse_weight: float = 0.15,
    pair_margin: float = 1e-5,
    max_pairs: int = 4096,
) -> "object":
    """
    RankNet on |split − target|: penalize wrong ordering of predicted |err|.

    Pair (i, j) is active when true |err_i| < |err_j| − margin (i closer to target).
    Loss uses softplus(err_pred_i − err_pred_j) on those pairs, weighted by |Δ relevance|
    (LambdaRank-lite: larger true gaps matter more).
    """
    import torch
    import torch.nn.functional as F

    pred = pred.reshape(-1)
    y = y.reshape(-1)
    err_true = torch.abs(y - target_split_ratio)
    err_pred = torch.abs(pred - target_split_ratio)
    mse = F.mse_loss(pred, y)

    n = pred.shape[0]
    if n < 2:
        return mse

    # [i,j]: j is worse than i (higher true error)
    gap = err_true.unsqueeze(1) - err_true.unsqueeze(0)
    mask = gap > pair_margin
    if not mask.any():
        return mse_weight * mse

    diff_pred = err_pred.unsqueeze(0) - err_pred.unsqueeze(1)
    pair = F.softplus(diff_pred)[mask]
    weights = gap[mask].detach()
    w_sum = weights.sum().clamp(min=1e-8)

    if int(mask.sum()) > max_pairs:
        idx = torch.where(mask)
        k = idx[0].shape[0]
        sel = torch.randint(0, k, (max_pairs,), device=pred.device)
        pair_vals = F.softplus(diff_pred[idx[0][sel], idx[1][sel]])
        w_sel = weights[idx[0][sel], idx[1][sel]]
        rank_term = (pair_vals * w_sel).sum() / max_pairs
    else:
        rank_term = (pair * weights).sum() / w_sum

    return mse_weight * mse + (1.0 - mse_weight) * rank_term
