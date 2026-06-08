#!/usr/bin/env python3
"""Ablation: selection policies on a fixed labeled proposal pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nano_inv.manifold import EBeamManifold  # noqa: E402
from nano_inv.surrogate import (  # noqa: E402
    append_sigma_feature,
    build_labeled_table,
    drop_invalid_targets,
    load_artifact,
    load_mask_feature_matrix,
    make_mlp_pipeline,
    surrogate_ranking_scores,
)

OUT = REPO / "data/phase1/release"
DEFAULT_CFG = REPO / "configs/wedge_a.yaml"
TARGET = 0.5
TOL = 0.05
TOP_KS = (20, 50)
SEED = 42
RANK_SUR = REPO / "data/phase1/wedge_a/surrogate_improved_rank/surrogate.joblib"
REG_SUR = REPO / "data/phase1/wedge_a/surrogate_improved/surrogate.joblib"


def filter_existing_latents(table: pd.DataFrame) -> pd.DataFrame:
    if "latent_path" not in table.columns:
        return table
    ok = [bool((REPO / p).exists()) for p in table["latent_path"].astype(str)]
    return table.loc[ok].reset_index(drop=True)


def resolve_corpus(cfg: dict) -> Path:
    merged = REPO / "data/phase0/sim_results_phase0_v1_all.csv"
    phase0 = REPO / cfg["data"]["sim_corpus"]
    return merged if merged.exists() else phase0


def pool_from_corpus(cfg: dict, *, source_filter: str, fast: bool) -> pd.DataFrame:
    manifest = REPO / cfg["data"]["manifest"]
    sim = resolve_corpus(cfg)
    labeled = build_labeled_table(
        REPO, manifest, sim, recipe_version="phase0_v1", source_filter=source_filter
    )
    labeled = drop_invalid_targets(labeled, "split_ratio_upper")
    labeled = filter_existing_latents(labeled)
    if fast and len(labeled) > 500:
        labeled = labeled.sample(n=500, random_state=SEED).reset_index(drop=True)
    return labeled


def predict_mse(pipe, X: np.ndarray) -> np.ndarray:
    return pipe.predict(X)


def policy_metrics(
    err_true: np.ndarray,
    scores: np.ndarray,
    *,
    k: int,
    policy: str,
) -> dict:
    k = min(k, len(err_true))
    idx = np.argpartition(scores, k - 1)[:k]
    sel_err = err_true[idx]
    in_spec = float((sel_err <= TOL).mean())
    mean_err = float(sel_err.mean())
    rng = np.random.default_rng(SEED + k)
    rand_idx = rng.choice(len(err_true), size=k, replace=False)
    rand_err = err_true[rand_idx]
    rand_mean = float(rand_err.mean())
    enrichment = (rand_mean - mean_err) / rand_mean if rand_mean > 0 else 0.0
    return {
        "policy": policy,
        "k": k,
        "mean_abs_split_error": mean_err,
        "fraction_in_spec": in_spec,
        "random_mean_abs_error": rand_mean,
        "enrichment_vs_random": enrichment,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--pool-csv", type=Path, default=None, help="Ranked candidates CSV with labels")
    p.add_argument("--source-filter", default="perturb_plus_search")
    p.add_argument("--rank-surrogate", type=Path, default=RANK_SUR)
    p.add_argument("--reg-surrogate", type=Path, default=REG_SUR)
    p.add_argument("--train-mse", action="store_true", help="Train fresh MSE MLP on pool")
    p.add_argument("--fast", action="store_true", help="Subsample corpus pool to 500 rows")
    args = p.parse_args()

    cfg_path = REPO / args.config if not args.config.is_absolute() else args.config
    cfg = yaml.safe_load(cfg_path.read_text())

    if args.pool_csv:
        pool = pd.read_csv(REPO / args.pool_csv if not args.pool_csv.is_absolute() else args.pool_csv)
        pool = drop_invalid_targets(pool, "split_ratio_upper")
        pool = filter_existing_latents(pool)
        if args.fast and len(pool) > 500:
            pool = pool.sample(n=500, random_state=SEED).reset_index(drop=True)
    else:
        pool = pool_from_corpus(cfg, source_filter=args.source_filter, fast=args.fast)

    split = pool["split_ratio_upper"].astype(float).to_numpy()
    err_true = np.abs(split - TARGET)

    manifold = EBeamManifold.load()
    X = load_mask_feature_matrix(REPO, pool, pool=6, manifold=manifold, decode_from_latent=True)
    X = append_sigma_feature(X, pool)

    rank_path = args.rank_surrogate if args.rank_surrogate.is_absolute() else REPO / args.rank_surrogate
    rank_art = load_artifact(rank_path) if rank_path.exists() else None
    if rank_art is None:
        raise FileNotFoundError(f"rank surrogate not found: {rank_path}")

    rank_pred = np.array(
        [
            rank_art.predict_from_latent(
                np.load(REPO / lp),
                manifold=manifold,
                sigma=float(row["sigma"]) if "sigma" in row and pd.notna(row["sigma"]) else None,
            )
            for _, row in pool.iterrows()
            for lp in [row["latent_path"]]
        ]
    )
    rank_scores = surrogate_ranking_scores(
        rank_pred,
        target=rank_art.target,
        target_split_ratio=float(getattr(rank_art, "target_split_ratio", TARGET)),
    )

    if args.train_mse:
        mse_pipe = make_mlp_pipeline(hidden_layer_sizes=(256, 128, 64), max_iter=800, random_state=SEED)
        mse_pipe.fit(X, split)
        mse_pred = predict_mse(mse_pipe, X)
    else:
        reg_path = args.reg_surrogate if args.reg_surrogate.is_absolute() else REPO / args.reg_surrogate
        reg_art = load_artifact(reg_path)
        mse_pred = np.array(
            [
                reg_art.predict_from_latent(
                    np.load(REPO / lp),
                    manifold=manifold,
                    sigma=float(row["sigma"]) if "sigma" in row and pd.notna(row["sigma"]) else None,
                )
                for _, row in pool.iterrows()
                for lp in [row["latent_path"]]
            ]
        )
    mse_scores = np.abs(mse_pred - TARGET)
    oracle_scores = err_true.copy()
    rng = np.random.default_rng(SEED)
    random_scores = rng.random(len(err_true))

    policies = [
        ("random_topk", random_scores),
        ("mse_surrogate_topk", mse_scores),
        ("rank_surrogate_topk", rank_scores),
        ("oracle_topk", oracle_scores),
    ]

    rows = []
    for policy, scores in policies:
        for k in TOP_KS:
            rows.append(policy_metrics(err_true, scores, k=k, policy=policy))

    payload = {
        "pool_n": len(pool),
        "pool_source": str(args.pool_csv) if args.pool_csv else resolve_corpus(cfg).relative_to(REPO).as_posix(),
        "train_mse_fresh": args.train_mse,
        "fast_subsample": args.fast,
        "policies": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ablation_proposal_pool.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Proposal-pool selection ablation",
        "",
        f"Pool n={len(pool)} · K ∈ {list(TOP_KS)} · in-spec ≤ {TOL}",
        "",
        "| Policy | K | mean |err| | frac in-spec | enrichment vs random |",
        "|--------|---|-------------|--------------|----------------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['policy']} | {r['k']} | {r['mean_abs_split_error']:.4f} | "
            f"{r['fraction_in_spec']:.2%} | {r['enrichment_vs_random']:+.1%} |"
        )
    md = OUT / "ablation_proposal_pool.md"
    md.write_text("\n".join(lines) + "\n")
    print(f"wrote {md}")
    for r in rows:
        if r["k"] == 20:
            print(f"{r['policy']:22s} K=20 err={r['mean_abs_split_error']:.4f} enrich={r['enrichment_vs_random']:+.1%}")


if __name__ == "__main__":
    main()
