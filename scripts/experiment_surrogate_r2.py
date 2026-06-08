#!/usr/bin/env python3
"""
Train several surrogate variants on the merged corpus and compare val R² + ranking.

Fast (CPU only). Outputs: data/phase1/wedge_a/r2_experiments/summary.md + summary.json

Usage:
  PYTHONPATH=src .venv/bin/python scripts/experiment_surrogate_r2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.surrogate import (  # noqa: E402
    apply_training_target,
    build_labeled_table,
    drop_invalid_targets,
    save_artifact,
    train_surrogate_bundle,
)

SIM = REPO_ROOT / "data/phase0/sim_results_phase0_v1_all.csv"
MANIFEST = REPO_ROOT / "data/phase0/manifest.csv"
OUT_ROOT = REPO_ROOT / "data/phase1/wedge_a/r2_experiments"
TARGET_SPLIT = 0.5
TOP_K = 20
SEED = 42

EXPERIMENTS = [
    {
        "id": "baseline_improved",
        "architecture": "mask_mlp",
        "source_filter": "perturb_plus_search",
        "target": "split_ratio_upper",
        "sigma_feature": True,
        "sample_weight_mode": None,
        "hidden": (256, 128, 64),
        "max_iter": 800,
    },
    {
        "id": "perturb_only",
        "architecture": "mask_mlp",
        "source_filter": "perturb",
        "target": "split_ratio_upper",
        "sigma_feature": True,
        "sample_weight_mode": None,
        "hidden": (256, 128, 64),
        "max_iter": 800,
    },
    {
        "id": "abs_split_error",
        "architecture": "mask_mlp",
        "source_filter": "perturb_plus_search",
        "target": "abs_split_error",
        "sigma_feature": True,
        "sample_weight_mode": None,
        "hidden": (256, 128, 64),
        "max_iter": 800,
    },
    {
        "id": "latent_mlp",
        "architecture": "latent_mlp",
        "source_filter": "perturb_plus_search",
        "target": "split_ratio_upper",
        "sigma_feature": False,
        "sample_weight_mode": None,
        "hidden": (256, 128, 64),
        "max_iter": 800,
    },
    {
        "id": "in_spec_weight3",
        "architecture": "mask_mlp",
        "source_filter": "perturb_plus_search",
        "target": "split_ratio_upper",
        "sigma_feature": True,
        "sample_weight_mode": "in_spec_boost",
        "hidden": (256, 128, 64),
        "max_iter": 800,
    },
    {
        "id": "soft_target_weight",
        "architecture": "mask_mlp",
        "source_filter": "perturb_plus_search",
        "target": "split_ratio_upper",
        "sigma_feature": True,
        "sample_weight_mode": "soft_target",
        "hidden": (256, 128, 64),
        "max_iter": 800,
    },
    {
        "id": "bimodal_weight",
        "architecture": "mask_mlp",
        "source_filter": "perturb_plus_search",
        "target": "split_ratio_upper",
        "sigma_feature": True,
        "sample_weight_mode": "bimodal",
        "hidden": (256, 128, 64),
        "max_iter": 800,
    },
    {
        "id": "shallow_wide",
        "architecture": "mask_mlp",
        "source_filter": "perturb_plus_search",
        "target": "split_ratio_upper",
        "sigma_feature": True,
        "sample_weight_mode": None,
        "hidden": (512, 512),
        "max_iter": 1000,
    },
]


def ranking_eval_on_table(
    repo_root: Path,
    labeled: pd.DataFrame,
    artifact_dir: Path,
    *,
    train_target: str,
) -> dict:
    from nano_inv.manifold import EBeamManifold
    from nano_inv.surrogate import load_artifact, surrogate_ranking_scores

    artifact = load_artifact(artifact_dir / "surrogate.joblib")
    manifold = EBeamManifold.load()
    preds = []
    for _, row in labeled.iterrows():
        sig = row["sigma"] if "sigma" in row and pd.notna(row["sigma"]) else None
        preds.append(
            artifact.predict_from_latent(
                np.load(repo_root / row["latent_path"]),
                manifold=manifold,
                sigma=float(sig) if sig is not None else None,
            )
        )
    yhat = np.array(preds)

    split = labeled["split_ratio_upper"].astype(float).to_numpy()
    err_true = abs(split - TARGET_SPLIT)
    err_pred = surrogate_ranking_scores(
        yhat,
        target=artifact.target,
        target_split_ratio=float(getattr(artifact, "target_split_ratio", TARGET_SPLIT)),
    )

    order_s = err_pred.argsort()[:TOP_K]
    order_r = err_true.argsort()[:TOP_K]
    n_in_spec = int((err_true <= 0.05).sum())
    return {
        "n_samples": len(labeled),
        "n_in_spec_corpus": n_in_spec,
        "mean_abs_err_surrogate_topk": float(err_pred[order_s].mean()),
        "mean_abs_err_random_topk": float(err_true[order_r].mean()),
        "ranking_wins": bool(err_pred[order_s].mean() < err_true[order_r].mean()),
        "n_in_spec_surrogate_topk": int((err_true[order_s] <= 0.05).sum()),
        "n_in_spec_random_topk": int((err_true[order_r] <= 0.05).sum()),
    }


def run_one(exp: dict) -> dict:
    out_dir = OUT_ROOT / exp["id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled = build_labeled_table(
        REPO_ROOT,
        MANIFEST,
        SIM,
        recipe_version="phase0_v1",
        source_filter=exp["source_filter"],
    )
    labeled = drop_invalid_targets(labeled, "split_ratio_upper")
    labeled, train_target = apply_training_target(
        labeled, exp["target"], target_split_ratio=TARGET_SPLIT
    )
    labeled = drop_invalid_targets(labeled, train_target)

    artifact = train_surrogate_bundle(
        REPO_ROOT,
        labeled,
        architecture=exp["architecture"],
        target=exp["target"],
        holdout_fraction=0.2,
        seed=SEED,
        hidden_layer_sizes=exp["hidden"],
        max_iter=exp["max_iter"],
        source_filter=exp["source_filter"],
        mask_pool=6,
        target_split_ratio=TARGET_SPLIT,
        sigma_feature=exp["sigma_feature"],
        sample_weight_mode=exp.get("sample_weight_mode"),
    )
    save_artifact(artifact, out_dir)

    rank = ranking_eval_on_table(REPO_ROOT, labeled, out_dir, train_target=train_target)
    m = artifact.metrics.to_dict()
    row = {
        "id": exp["id"],
        "n_train": m["n_train"],
        "n_val": m["n_val"],
        "val_r2": m["val_r2"],
        "val_mae": m["val_mae"],
        "val_spearman_abs_err": m.get("val_spearman_abs_err"),
        "source_filter": exp["source_filter"],
        "target": exp["target"],
        "architecture": exp["architecture"],
        "sample_weight_mode": exp.get("sample_weight_mode"),
        **rank,
    }
    (out_dir / "experiment_row.json").write_text(json.dumps(row, indent=2))
    print(
        f"{exp['id']:22s}  n={row['n_train']:4d}  R²={row['val_r2']:7.3f}  "
        f"rank_wins={row['ranking_wins']}  top20_in_spec={row['n_in_spec_surrogate_topk']}"
    )
    return row


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [run_one(e) for e in EXPERIMENTS]
    rows.sort(key=lambda r: r["val_r2"], reverse=True)

    summary_path = OUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))

    lines = [
        "# Surrogate R² experiments",
        "",
        f"Corpus: `{SIM.relative_to(REPO_ROOT)}` · holdout 20% · seed {SEED}",
        "",
        "| id | n_train | val_r2 | val_mae | ranking_wins | top20 in-spec |",
        "|----|---------|--------|---------|--------------|---------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['n_train']} | {r['val_r2']:.3f} | {r['val_mae']:.3f} | "
            f"{r['ranking_wins']} | {r['n_in_spec_surrogate_topk']} |"
        )
    lines.extend(
        [
            "",
            "**Pick:** highest `val_r2` that keeps `ranking_wins=true`.",
            "",
            "Re-run: `PYTHONPATH=src .venv/bin/python scripts/experiment_surrogate_r2.py`",
        ]
    )
    (OUT_ROOT / "summary.md").write_text("\n".join(lines))
    print(f"\nWrote {summary_path} and {OUT_ROOT / 'summary.md'}")
    best = rows[0]
    print(f"Best R²: {best['id']} ({best['val_r2']:.3f})")


if __name__ == "__main__":
    main()
