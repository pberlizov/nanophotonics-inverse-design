"""Tests for rank_loss (numpy path; torch optional)."""

from __future__ import annotations

import numpy as np

from nano_inv.rank_loss import rank_weighted_regression_weights


def test_rank_weights_monotone() -> None:
    y = np.array([0.5, 0.65, 0.42, 0.9])
    w = rank_weighted_regression_weights(y, target_split_ratio=0.5)
    assert w[0] > w[1]  # err 0 vs 0.15
    assert w[2] > w[1]  # err 0.08 vs 0.15
    assert w[0] > w[3]  # err 0 vs 0.4


def test_pairwise_loss_prefers_correct_order() -> None:
    pytest = __import__("pytest")
    torch = pytest.importorskip("torch")
    from nano_inv.rank_loss import pairwise_rank_loss_torch

    target = 0.5
    y = torch.tensor([0.48, 0.70, 0.51, 0.80])
    good_pred = torch.tensor([0.49, 0.75, 0.52, 0.85])
    bad_pred = torch.tensor([0.80, 0.49, 0.75, 0.48])
    loss_good = float(pairwise_rank_loss_torch(good_pred, y, target_split_ratio=target))
    loss_bad = float(pairwise_rank_loss_torch(bad_pred, y, target_split_ratio=target))
    assert loss_good < loss_bad
