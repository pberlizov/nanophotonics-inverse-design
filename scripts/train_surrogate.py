#!/usr/bin/env python3
"""Train Phase 0 surrogate (latent MLP, pooled-mask MLP, or mask CNN)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.surrogate import (  # noqa: E402
    apply_training_target,
    build_labeled_table,
    drop_invalid_targets,
    save_artifact,
    train_surrogate_bundle,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase0.yaml")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--sim-results", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument(
        "--target",
        type=str,
        default="split_ratio_upper",
        help="split_ratio_upper | abs_split_error (needs --target-split-ratio)",
    )
    p.add_argument("--target-split-ratio", type=float, default=0.5)
    p.add_argument("--min-ok", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-fraction", type=float, default=None)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--hidden", type=str, default="128,64")
    p.add_argument(
        "--architecture",
        type=str,
        choices=["latent_mlp", "mask_mlp", "mask_cnn"],
        default=None,
        help="Override configs/phase0.yaml surrogate.architecture",
    )
    p.add_argument(
        "--sources",
        type=str,
        default=None,
        help="all | perturb | perturb_plus_search | perlin (override config source_filter)",
    )
    p.add_argument("--mask-pool", type=int, default=None, help="Max-pool factor for mask_mlp")
    p.add_argument("--cnn-epochs", type=int, default=40)
    p.add_argument(
        "--near-target-max-abs-err",
        type=float,
        default=None,
        help="Train only on rows with |split-target| <= this (e.g. 0.15)",
    )
    p.add_argument(
        "--sigma-feature",
        action="store_true",
        help="Append sigma column to mask_mlp features",
    )
    p.add_argument(
        "--sample-weight-mode",
        type=str,
        default=None,
        help="in_spec_boost | soft_target | bimodal",
    )
    p.add_argument(
        "--sample-weight-in-spec-tol",
        type=float,
        default=0.05,
        help="|split-0.5| threshold for in_spec_boost",
    )
    p.add_argument(
        "--champion-weight",
        type=float,
        default=2.0,
        help="Multiply weight for champion latents from config champions.latent_paths",
    )
    p.add_argument(
        "--decode-masks-from-latent",
        action="store_true",
        help="Hard-decode latent via manifold (not mask_path files)",
    )
    p.add_argument(
        "--champion-latent",
        action="append",
        default=None,
        help="Latent path(s) for champion sample-weight boost (repeatable)",
    )
    p.add_argument(
        "--loss-mode",
        type=str,
        choices=["regression", "pairwise_rank", "rank_weighted"],
        default=None,
        help="regression (MSE) | pairwise_rank (torch RankNet) | rank_weighted (sklearn)",
    )
    p.add_argument(
        "--rank-mse-weight",
        type=float,
        default=0.15,
        help="MSE fraction in pairwise_rank loss (rest is rank pairs)",
    )
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = REPO_ROOT / cfg["data"]["root"]
    manifest_path = Path(args.manifest) if args.manifest else REPO_ROOT / cfg["data"]["manifest"]
    sim_path = Path(args.sim_results) if args.sim_results else data_root / "sim_results.csv"
    out_dir = Path(args.output_dir) if args.output_dir else data_root / "surrogate"

    sur_cfg = cfg.get("surrogate") or {}
    holdout = (
        args.holdout_fraction
        if args.holdout_fraction is not None
        else float(sur_cfg.get("holdout_fraction", 0.2))
    )
    architecture = args.architecture or sur_cfg.get("architecture", "latent_mlp")
    if architecture == "mlp":
        architecture = "latent_mlp"
    source_filter = args.sources or sur_cfg.get("source_filter", "all")
    mask_pool = args.mask_pool or int(sur_cfg.get("mask_pool", 6))
    recipe_version = (cfg.get("meep") or {}).get("recipe_version")

    if not sim_path.exists():
        raise SystemExit(f"sim results not found: {sim_path}")

    labeled = build_labeled_table(
        REPO_ROOT,
        manifest_path,
        sim_path,
        recipe_version=recipe_version,
        source_filter=source_filter,
    )
    n_ok_raw = len(labeled)
    labeled = drop_invalid_targets(labeled, "split_ratio_upper")
    labeled, train_target = apply_training_target(
        labeled, args.target, target_split_ratio=float(args.target_split_ratio)
    )
    labeled = drop_invalid_targets(labeled, train_target)
    n_ok = len(labeled)
    if n_ok < args.min_ok:
        raise SystemExit(f"only {n_ok} ok rows (need --min-ok {args.min_ok})")

    hidden = tuple(int(x) for x in args.hidden.split(","))
    near_err = args.near_target_max_abs_err
    if near_err is None and sur_cfg.get("near_target_max_abs_err") is not None:
        near_err = float(sur_cfg["near_target_max_abs_err"])
    sigma_feature = args.sigma_feature or bool(sur_cfg.get("sigma_feature", False))
    sample_weight_mode = args.sample_weight_mode or sur_cfg.get("sample_weight_mode")
    sw_tol = float(
        sur_cfg.get("sample_weight_in_spec_tol", args.sample_weight_in_spec_tol)
    )
    champion_weight = float(sur_cfg.get("champion_weight", args.champion_weight))
    champ_paths = args.champion_latent or (cfg.get("champions") or {}).get("latent_paths")
    decode_latent = args.decode_masks_from_latent or bool(
        sur_cfg.get("decode_masks_from_latent", False)
    )
    loss_mode = args.loss_mode or sur_cfg.get("loss_mode", "regression")
    rank_mse_weight = float(sur_cfg.get("rank_mse_weight", args.rank_mse_weight))
    manifold = None
    if decode_latent:
        from nano_inv.manifold import EBeamManifold

        manifold = EBeamManifold.load()

    artifact = train_surrogate_bundle(
        REPO_ROOT,
        labeled,
        architecture=architecture,  # type: ignore[arg-type]
        target=args.target,
        holdout_fraction=holdout,
        seed=args.seed,
        hidden_layer_sizes=hidden,
        max_iter=args.max_iter,
        source_filter=source_filter,
        mask_pool=mask_pool,
        cnn_epochs=args.cnn_epochs,
        target_split_ratio=float(args.target_split_ratio),
        near_target_max_abs_err=near_err,
        sigma_feature=sigma_feature,
        sample_weight_mode=sample_weight_mode,
        sample_weight_in_spec_tol=sw_tol,
        champion_latent_paths=champ_paths,
        champion_weight=champion_weight,
        decode_masks_from_latent=decode_latent,
        manifold=manifold,
        loss_mode=str(loss_mode),
        rank_mse_weight=rank_mse_weight,
    )
    artifact.recipe_version = recipe_version
    save_artifact(artifact, out_dir)

    labeled[["sample_id", "source", "latent_path", "mask_path", train_target]].to_csv(
        out_dir / "training_rows.csv", index=False
    )

    try:
        sim_rel = str(sim_path.relative_to(REPO_ROOT))
    except ValueError:
        sim_rel = str(sim_path)
    try:
        out_rel = str(out_dir.relative_to(REPO_ROOT))
    except ValueError:
        out_rel = str(out_dir)

    summary = {
        **artifact.metrics.to_dict(),
        "output_dir": out_rel,
        "n_ok_labeled": n_ok,
        "n_ok_dropped_invalid_target": n_ok_raw - n_ok,
        "sim_results": sim_rel,
        "architecture": architecture,
        "source_filter": source_filter,
        "mask_pool": mask_pool,
        "loss_mode": loss_mode,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
