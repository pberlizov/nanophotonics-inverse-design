"""Latent search helpers for Phase 0 (surrogate-guided BO)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import optuna

from nano_inv.drc_heuristic import check_mask_heuristic
from nano_inv.latent import pad_latent_to_standard, sample_latent_perturbation
from nano_inv.search_objectives import SearchConfig, SearchMode, split_ratio_loss
from nano_inv.surrogate import SurrogateArtifact

if TYPE_CHECKING:
    from nano_inv.manifold import EBeamManifold

__all__ = ["SearchConfig", "SearchMode", "split_ratio_loss", "sample_latent_from_trial", "evaluate_candidate", "make_objective"]


def sample_latent_from_trial(
    trial: optuna.Trial,
    *,
    reference: np.ndarray,
    rng: np.random.Generator,
    mode: SearchMode,
) -> tuple[np.ndarray, dict]:
    meta: dict = {}
    if mode == "mixed":
        mode = trial.suggest_categorical("family", ["perturb", "perlin"])

    if mode == "perturb":
        sigma = trial.suggest_float("sigma", 0.02, 0.20, log=True)
        z = sample_latent_perturbation(reference, rng, sigma=sigma)
        meta = {"family": "perturb", "sigma": sigma}
    elif mode == "perlin":
        from nano_inv.manifold import sample_latent_perlin

        scale = trial.suggest_float("scale", 2.5, 5.0)
        offset_x = trial.suggest_float("offset_x", 0.0, 30.0)
        offset_y = trial.suggest_float("offset_y", 0.0, 30.0)
        dim = trial.suggest_int("dim", 2, 4)
        z = sample_latent_perlin(scale=scale, offset=(offset_x, offset_y), dim=dim)
        meta = {
            "family": "perlin",
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "dim": dim,
        }
    else:
        raise ValueError(f"unknown search mode {mode!r}")

    return z, meta


def evaluate_candidate(
    z: np.ndarray,
    *,
    surrogate: SurrogateArtifact,
    manifold: EBeamManifold,
    cfg: SearchConfig,
) -> tuple[float, float, bool, dict]:
    z_std = pad_latent_to_standard(z)
    pred = surrogate.predict_from_latent(z_std, manifold=manifold)
    loss = split_ratio_loss(pred, cfg.target_split_ratio)
    drc_pass = True
    drc_meta: dict = {}
    if cfg.require_drc:
        mask = manifold.decode_numpy(z)
        drc = check_mask_heuristic(mask)
        drc_pass = drc.passed
        drc_meta = {
            "drc_heuristic_pass": drc.passed,
            "fill_ratio": drc.fill_ratio,
            "min_run_length": drc.min_run_length,
            "drc_reasons": ";".join(drc.reasons),
        }
        if not drc_pass:
            loss += cfg.drc_penalty
    in_spec = abs(pred - cfg.target_split_ratio) <= cfg.tolerance
    return loss, pred, in_spec, drc_meta


def make_objective(
    *,
    surrogate: SurrogateArtifact,
    manifold: EBeamManifold,
    reference: np.ndarray,
    seed: int,
    cfg: SearchConfig,
):
    def objective(trial: optuna.Trial) -> float:
        trial_rng = np.random.default_rng(seed + trial.number)
        z, meta = sample_latent_from_trial(
            trial, reference=reference, rng=trial_rng, mode=cfg.mode
        )
        loss, pred, in_spec, drc_meta = evaluate_candidate(
            z, surrogate=surrogate, manifold=manifold, cfg=cfg
        )
        trial.set_user_attr("pred_split_ratio_upper", pred)
        trial.set_user_attr("in_spec", in_spec)
        for k, v in meta.items():
            trial.set_user_attr(k, v)
        for k, v in drc_meta.items():
            trial.set_user_attr(k, v)
        return loss

    return objective
