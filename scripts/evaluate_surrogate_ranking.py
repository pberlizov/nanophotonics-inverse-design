#!/usr/bin/env python3
"""Evaluate whether a surrogate ranks masks better than random (MEEP ground truth)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.manifest import filter_by_source  # noqa: E402
from nano_inv.surrogate import (  # noqa: E402
    build_labeled_table,
    drop_invalid_targets,
    load_artifact,
    surrogate_ranking_scores,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase0.yaml")
    p.add_argument("--surrogate", type=Path, required=True)
    p.add_argument("--sim-results", type=Path, default=None)
    p.add_argument("--sources", type=str, default="perturb")
    p.add_argument("--target", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def mean_abs_err(y: np.ndarray, target: float) -> float:
    return float(np.mean(np.abs(y - target)))


def topk_indices(scores: np.ndarray, k: int, *, best: str = "low") -> np.ndarray:
    k = min(k, len(scores))
    if best == "low":
        return np.argpartition(scores, k - 1)[:k]
    return np.argpartition(-scores, k - 1)[:k]


def main() -> None:
    args = parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    data_root = REPO_ROOT / cfg["data"]["root"]
    sim_path = args.sim_results or (data_root / "sim_results_phase0_v1.csv")
    sur_path = args.surrogate
    if not sur_path.is_absolute():
        sur_path = REPO_ROOT / sur_path

    artifact = load_artifact(sur_path / "surrogate.joblib")
    manifest = REPO_ROOT / cfg["data"]["manifest"]
    labeled = build_labeled_table(REPO_ROOT, manifest, sim_path)
    labeled = filter_by_source(labeled, args.sources)
    labeled = drop_invalid_targets(labeled, "split_ratio_upper")

    from nano_inv.manifold import EBeamManifold

    manifold = EBeamManifold.load()
    y_true = labeled["split_ratio_upper"].to_numpy()
    y_pred = np.array(
        [artifact.predict_from_latent(np.load(REPO_ROOT / p), manifold=manifold) for p in labeled["latent_path"]]
    )

    err_true = np.abs(y_true - args.target)
    t_split = float(getattr(artifact, "target_split_ratio", args.target))
    err_pred = surrogate_ranking_scores(
        y_pred, target=artifact.target, target_split_ratio=t_split
    )
    rho, pval = spearmanr(err_true, err_pred)

    rng = np.random.default_rng(args.seed)
    n = len(labeled)
    k = args.top_k
    sur_idx = topk_indices(err_pred, k, best="low")
    rand_idx = rng.choice(n, size=k, replace=False)

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(REPO_ROOT))
        except ValueError:
            return str(p.resolve())

    metrics = {
        "n_samples": n,
        "target_split_ratio": args.target,
        "top_k": k,
        "surrogate": _rel(sur_path),
        "sim_results": _rel(sim_path),
        "spearman_err": float(rho) if np.isfinite(rho) else None,
        "spearman_pvalue": float(pval) if np.isfinite(pval) else None,
        "val_mae_pred": float(np.mean(np.abs(y_true - y_pred))),
        "mean_abs_err_surrogate_topk": mean_abs_err(y_true[sur_idx], args.target),
        "mean_abs_err_random_topk": mean_abs_err(y_true[rand_idx], args.target),
        "ranking_wins": mean_abs_err(y_true[sur_idx], args.target)
        < mean_abs_err(y_true[rand_idx], args.target),
        "n_in_spec_surrogate_topk": int(np.sum(np.abs(y_true[sur_idx] - args.target) <= 0.05)),
        "n_in_spec_random_topk": int(np.sum(np.abs(y_true[rand_idx] - args.target) <= 0.05)),
        "architecture": getattr(artifact, "architecture", "unknown"),
    }
    if (sur_path / "metrics.json").exists():
        metrics["train_metrics"] = json.loads((sur_path / "metrics.json").read_text())

    out = args.output or (data_root / "surrogate_ranking_eval.json")
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
