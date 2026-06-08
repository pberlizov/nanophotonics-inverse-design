#!/usr/bin/env python3
"""
Deep-work R² experiments: group splits, target transforms, joint rank+MSE.

Logs append to data/phase1/wedge_a/r2_deep_work_log.md

  PYTHONPATH=src .venv/bin/python scripts/r2_deep_work.py
  PYTHONPATH=src .venv/bin/python scripts/r2_deep_work.py --only exp_group_source
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nano_inv.manifold import EBeamManifold  # noqa: E402
from nano_inv.surrogate import (  # noqa: E402
    apply_training_target,
    append_sigma_feature,
    build_labeled_table,
    compute_sample_weights,
    drop_invalid_targets,
    filter_near_target,
    load_artifact,
    load_mask_feature_matrix,
    make_mlp_pipeline,
    save_artifact,
    surrogate_ranking_scores,
    train_rank_mlp,
    train_sklearn_surrogate,
    _metrics_from_predictions,
)

SIM = REPO / "data/phase0/sim_results_phase0_v1_all.csv"
MANIFEST = REPO / "data/phase0/manifest.csv"
LOG = REPO / "data/phase1/wedge_a/r2_deep_work_log.md"
OUT_ROOT = REPO / "data/phase1/wedge_a/r2_deep_work"
TARGET_SPLIT = 0.5
TOP_K = 20
SEED = 42
HOLDOUT = 0.2

CHAMPION_LATENTS = [
    "data/phase0/latents/ref_published_latent.npy",
    "data/phase1/meep_search_local/candidates/latents/local_00022_latent.npy",
    "data/phase1/meep_search_deep/candidates/latents/meep_bo_00128_latent.npy",
    "data/phase1/wedge_a/meep_gated_shortlist/latents/cand_000160_latent.npy",
    "data/phase1/wedge_a/meep_gated_shortlist/round_rank/latents/cand_000261_latent.npy",
]


def extract_group(sample_id: str, source: str) -> str:
    if sample_id.startswith("sig_"):
        m = re.match(r"(sig_\d+)", sample_id)
        if m:
            return m.group(1)
    if sample_id.startswith("cand_"):
        return sample_id.rsplit("_", 1)[0] if "_" in sample_id else sample_id
    return source or "unknown"


def group_split_indices(groups: np.ndarray, *, seed: int, holdout: float) -> tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
    idx = np.arange(len(groups))
    train_idx, val_idx = next(gss.split(idx, groups=groups))
    return train_idx, val_idx


def random_split_indices(n: int, *, seed: int, holdout: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(n * holdout))
    return idx[n_val:], idx[:n_val]


def ranking_eval(
    repo: Path,
    labeled: pd.DataFrame,
    artifact_dir: Path,
    *,
    train_target: str,
) -> dict:
    artifact = load_artifact(artifact_dir / "surrogate.joblib")
    manifold = EBeamManifold.load()
    preds = []
    for _, row in labeled.iterrows():
        sig = row["sigma"] if "sigma" in row and pd.notna(row["sigma"]) else None
        preds.append(
            artifact.predict_from_latent(
                np.load(repo / row["latent_path"]),
                manifold=manifold,
                sigma=float(sig) if sig is not None else None,
            )
        )
    yhat = np.array(preds)
    split = labeled["split_ratio_upper"].astype(float).to_numpy()
    err_true = np.abs(split - TARGET_SPLIT)
    err_pred = surrogate_ranking_scores(
        yhat,
        target=artifact.target,
        target_split_ratio=float(getattr(artifact, "target_split_ratio", TARGET_SPLIT)),
    )
    order_s = err_pred.argsort()[:TOP_K]
    order_r = err_true.argsort()[:TOP_K]
    return {
        "spearman_err": float(pd.Series(err_true).corr(pd.Series(err_pred), method="spearman")),
        "mean_abs_err_surrogate_topk": float(err_pred[order_s].mean()),
        "mean_abs_err_random_topk": float(err_true[order_r].mean()),
        "ranking_wins": bool(err_pred[order_s].mean() < err_true[order_r].mean()),
        "n_in_spec_surrogate_topk": int((err_true[order_s] <= 0.05).sum()),
        "n_in_spec_random_topk": int((err_true[order_r] <= 0.05).sum()),
    }


def train_with_split(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    target: str,
    weights: np.ndarray | None,
    loss_mode: str,
    hidden: tuple[int, ...],
    max_iter: int,
    rank_mse_weight: float,
) -> tuple[object, dict]:
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    w_train = weights[train_idx] if weights is not None else None

    if loss_mode == "pairwise_rank":
        # rank trainer does its own split — approximate by fitting on train only via temp hack:
        # use sklearn path on train, evaluate val manually with torch model from rank trainer
        pipe, _ = train_rank_mlp(
            X,
            y,
            target=target,
            holdout_fraction=HOLDOUT,
            seed=SEED,
            hidden_layer_sizes=hidden,
            max_iter=max_iter,
            architecture="mask_mlp",
            source_filter="perturb_plus_search",
            target_split_ratio=TARGET_SPLIT,
            mse_weight=rank_mse_weight,
        )
        pred_val = pipe.predict(X_val)
    else:
        pipe = make_mlp_pipeline(hidden_layer_sizes=hidden, max_iter=max_iter, random_state=SEED)
        if w_train is not None:
            pipe.fit(X_train, y_train, mlp__sample_weight=w_train)
        else:
            pipe.fit(X_train, y_train)
        pred_val = pipe.predict(X_val)

    metrics = _metrics_from_predictions(
        y_val,
        pred_val,
        target=target,
        n_total=len(X),
        n_train=len(train_idx),
        n_val=len(val_idx),
        holdout_fraction=HOLDOUT,
        seed=SEED,
        architecture="mask_mlp",
        source_filter="perturb_plus_search",
        target_split_ratio=TARGET_SPLIT,
    )
    # In-spec-only R² on val
    split_val = None  # caller may pass via closure — set below in run_exp
    row = metrics.to_dict()
    return pipe, row


def prepare_data(
    *,
    source_filter: str,
    target: str,
    near_max: float | None,
    exclude_sim_budget: bool,
) -> tuple[pd.DataFrame, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labeled = build_labeled_table(
        REPO, MANIFEST, SIM, recipe_version="phase0_v1", source_filter=source_filter
    )
    if exclude_sim_budget:
        labeled = labeled.loc[labeled["source"] != "sim_budget"].reset_index(drop=True)
    labeled = drop_invalid_targets(labeled, "split_ratio_upper")
    labeled = filter_near_target(
        labeled, target_split_ratio=TARGET_SPLIT, max_abs_err=near_max
    )
    labeled, train_target = apply_training_target(
        labeled, target, target_split_ratio=TARGET_SPLIT
    )
    labeled = drop_invalid_targets(labeled, train_target)

    manifold = EBeamManifold.load()
    X = load_mask_feature_matrix(
        REPO, labeled, pool=6, manifold=manifold, decode_from_latent=True
    )
    X = append_sigma_feature(X, labeled)
    y = labeled[train_target].astype(np.float64).to_numpy()
    groups = np.array(
        [extract_group(str(r["sample_id"]), str(r["source"])) for _, r in labeled.iterrows()]
    )
    split_col = labeled["split_ratio_upper"].astype(float).to_numpy()
    latent_col = labeled["latent_path"].astype(str).to_numpy()
    weights = compute_sample_weights(
        split_col,
        target_split_ratio=TARGET_SPLIT,
        mode="in_spec_boost",
        in_spec_tol=0.05,
        latent_paths=latent_col,
        champion_latent_paths=CHAMPION_LATENTS,
        champion_weight=2.0,
    )
    return labeled, train_target, X, y, groups, weights


def run_exp(exp: dict) -> dict:
    out_dir = OUT_ROOT / exp["id"]
    out_dir.mkdir(parents=True, exist_ok=True)

    labeled, train_target, X, y, groups, weights = prepare_data(
        source_filter=exp.get("source_filter", "perturb_plus_search"),
        target=exp["target"],
        near_max=exp.get("near_max"),
        exclude_sim_budget=exp.get("exclude_sim_budget", False),
    )

    split_mode = exp.get("split_mode", "random")
    if split_mode == "group_source":
        grp = labeled["source"].astype(str).to_numpy()
        train_idx, val_idx = group_split_indices(grp, seed=SEED, holdout=HOLDOUT)
    elif split_mode == "group_sig":
        train_idx, val_idx = group_split_indices(groups, seed=SEED, holdout=HOLDOUT)
    else:
        train_idx, val_idx = random_split_indices(len(X), seed=SEED, holdout=HOLDOUT)

    sw = weights if exp.get("in_spec_boost") else None
    pipe, m = train_with_split(
        X,
        y,
        train_idx,
        val_idx,
        target=train_target,
        weights=sw,
        loss_mode=exp.get("loss_mode", "regression"),
        hidden=tuple(exp.get("hidden", (256, 128, 64))),
        max_iter=int(exp.get("max_iter", 800)),
        rank_mse_weight=float(exp.get("rank_mse_weight", 0.15)),
    )

    # In-spec val R²
    split_val = labeled["split_ratio_upper"].astype(float).to_numpy()[val_idx]
    pred_val = pipe.predict(X[val_idx])
    in_spec = np.abs(split_val - TARGET_SPLIT) <= 0.05
    if in_spec.sum() >= 5:
        from sklearn.metrics import r2_score

        m["val_r2_in_spec"] = float(r2_score(split_val[in_spec], pred_val[in_spec]))
        m["n_val_in_spec"] = int(in_spec.sum())
    else:
        m["val_r2_in_spec"] = None
        m["n_val_in_spec"] = int(in_spec.sum())

    from nano_inv.surrogate import SurrogateArtifact, SurrogateMetrics

    sm = SurrogateMetrics(**{k: v for k, v in m.items() if k in SurrogateMetrics.__dataclass_fields__})
    artifact = SurrogateArtifact(
        pipeline=pipe,
        target=train_target,
        recipe_version=None,
        metrics=sm,
        architecture="mask_mlp",
        input_dim=X.shape[1],
        mask_pool=6,
        target_split_ratio=TARGET_SPLIT,
        sigma_feature=True,
        loss_mode=exp.get("loss_mode", "regression"),
    )
    save_artifact(artifact, out_dir)

    rank = ranking_eval(REPO, labeled, out_dir, train_target=train_target)
    row = {
        "id": exp["id"],
        "config": exp,
        "out_dir": str(out_dir.relative_to(REPO)),
        **m,
        **rank,
    }
    (out_dir / "experiment.json").write_text(json.dumps(row, indent=2))
    print(
        f"{exp['id']:28s}  n={m['n_train']+m['n_val']:4d}  "
        f"R²={m['val_r2']:7.3f}  R²_in={m.get('val_r2_in_spec')}  "
        f"ρ={m.get('val_spearman_abs_err', 0):.3f}  rank_wins={rank['ranking_wins']}"
    )
    return row


EXPERIMENTS = [
    {"id": "baseline_random", "split_mode": "random", "target": "split_ratio_upper"},
    {
        "id": "group_source",
        "split_mode": "group_source",
        "target": "split_ratio_upper",
        "in_spec_boost": True,
    },
    {
        "id": "group_sig",
        "split_mode": "group_sig",
        "target": "split_ratio_upper",
        "in_spec_boost": True,
    },
    {
        "id": "abs_err_group_sig",
        "split_mode": "group_sig",
        "target": "abs_split_error",
        "in_spec_boost": True,
    },
    {
        "id": "near_spec_015",
        "split_mode": "group_sig",
        "target": "split_ratio_upper",
        "near_max": 0.15,
        "in_spec_boost": True,
    },
    {
        "id": "no_sim_budget",
        "split_mode": "random",
        "target": "split_ratio_upper",
        "exclude_sim_budget": True,
        "in_spec_boost": True,
    },
    {
        "id": "rank_mse_joint",
        "split_mode": "group_sig",
        "target": "split_ratio_upper",
        "loss_mode": "pairwise_rank",
        "rank_mse_weight": 0.35,
        "in_spec_boost": True,
    },
    {
        "id": "wide_512",
        "split_mode": "group_sig",
        "target": "split_ratio_upper",
        "hidden": (512, 256, 128),
        "max_iter": 1000,
        "in_spec_boost": True,
    },
]


def append_log(rows: list[dict]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"\n## Run {ts}\n", "| id | n | val_r2 | val_r2_in_spec | spearman | ranking_wins | top20 in-spec | out_dir |", "|----|---|--------|----------------|----------|--------------|---------------|---------|"]
    for r in rows:
        r2in = r.get("val_r2_in_spec")
        r2in_s = f"{r2in:.3f}" if r2in is not None else "—"
        lines.append(
            f"| {r['id']} | {r['n_train']+r['n_val']} | {r['val_r2']:.3f} | {r2in_s} | "
            f"{r.get('val_spearman_abs_err', 0):.3f} | {r['ranking_wins']} | "
            f"{r['n_in_spec_surrogate_topk']} | `{r['out_dir']}` |"
        )
    if not LOG.exists():
        header = "# R² deep work log\n\nCorpus: sim_results_phase0_v1_all.csv · perturb_plus_search\n"
        LOG.write_text(header)
    with LOG.open("a") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", action="append", default=None, help="Run subset of experiment ids")
    args = p.parse_args()
    exps = EXPERIMENTS
    if args.only:
        exps = [e for e in EXPERIMENTS if e["id"] in args.only]
    rows = [run_exp(e) for e in exps]
    rows.sort(key=lambda r: r["val_r2"], reverse=True)
    summary = OUT_ROOT / "summary.json"
    summary.write_text(json.dumps(rows, indent=2))
    append_log(rows)
    print(f"\nBest R²: {rows[0]['id']} ({rows[0]['val_r2']:.3f})")
    print(f"Log: {LOG}")


if __name__ == "__main__":
    main()
