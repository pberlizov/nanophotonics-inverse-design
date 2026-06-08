#!/usr/bin/env python3
"""Three surrogate validation splits: random, σ-grouped, chronological."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nano_inv.manifold import EBeamManifold  # noqa: E402
from nano_inv.surrogate import (  # noqa: E402
    append_sigma_feature,
    apply_training_target,
    build_labeled_table,
    compute_sample_weights,
    drop_invalid_targets,
    load_mask_feature_matrix,
    surrogate_ranking_scores,
    train_rank_mlp,
)

OUT = REPO / "data/phase1/release"
DEFAULT_CFG = REPO / "configs/wedge_a.yaml"
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

# Validation sources: wedge_a MEEP-gated shortlist era (+ round_rank ingest tag).
VAL_SOURCES = frozenset(
    {
        "meep_gated_shortlist",
        "meep_gated_shortlist_rank",
    }
)

TRAIN_EXCLUDE_SOURCES = frozenset(
    {
        "meep_gated_shortlist",
        "meep_gated_shortlist_r2",
        "meep_gated_shortlist_r3",
        "meep_gated_shortlist_r4",
        "meep_gated_shortlist_r5",
        "meep_gated_shortlist_r6",
        "meep_gated_shortlist_r7",
        "meep_gated_shortlist_rank",
        "sim_budget",
    }
)


def filter_existing_latents(table: pd.DataFrame) -> pd.DataFrame:
    if "latent_path" not in table.columns:
        return table
    ok = [bool((REPO / p).exists()) for p in table["latent_path"].astype(str)]
    return table.loc[ok].reset_index(drop=True)


def extract_group(sample_id: str, source: str) -> str:
    if sample_id.startswith("sig_"):
        m = re.match(r"(sig_\d+)", sample_id)
        if m:
            return m.group(1)
    if sample_id.startswith("cand_"):
        return sample_id.rsplit("_", 1)[0] if "_" in sample_id else sample_id
    return source or "unknown"


def resolve_corpus(cfg: dict) -> Path:
    phase1 = REPO / "data/phase1/sim_results_phase0_v1_all.csv"
    phase0 = REPO / cfg["data"]["sim_corpus"]
    if not phase0.is_absolute():
        phase0 = REPO / phase0
    if phase1.exists() and phase1.stat().st_size > 1000:
        merged = REPO / "data/phase0/sim_results_phase0_v1_all.csv"
        if merged.exists():
            return merged
    return phase0


def prepare_features(
    labeled: pd.DataFrame,
    *,
    train_target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    manifold = EBeamManifold.load()
    X = load_mask_feature_matrix(REPO, labeled, pool=6, manifold=manifold, decode_from_latent=True)
    X = append_sigma_feature(X, labeled)
    y = labeled[train_target].astype(np.float64).to_numpy()
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
    return X, y, weights


def eval_val(
    pipe,
    X_val: np.ndarray,
    split_val: np.ndarray,
    *,
    train_target: str,
) -> dict:
    pred_val = pipe.predict(X_val)
    err_true = np.abs(split_val - TARGET_SPLIT)
    err_pred = surrogate_ranking_scores(
        pred_val, target=train_target, target_split_ratio=TARGET_SPLIT
    )
    abs_err_true = err_true
    abs_err_pred = pred_val if train_target == "abs_split_error" else err_pred

    val_r2 = float(r2_score(abs_err_true, abs_err_pred)) if len(abs_err_true) >= 3 else None
    spearman = float(pd.Series(err_true).corr(pd.Series(err_pred), method="spearman"))

    order_s = err_pred.argsort()[:TOP_K]
    order_r = err_true.argsort()[:TOP_K]
    return {
        "n_val": int(len(split_val)),
        "val_r2_abs_split_error": val_r2,
        "val_spearman_abs_err": spearman,
        "mean_abs_err_surrogate_topk": float(err_true[order_s].mean()),
        "mean_abs_err_random_topk": float(err_true[order_r].mean()),
        "ranking_wins": bool(err_true[order_s].mean() < err_true[order_r].mean()),
        "n_in_spec_surrogate_topk": int((err_true[order_s] <= 0.05).sum()),
        "n_in_spec_random_topk": int((err_true[order_r] <= 0.05).sum()),
    }


def run_regime(
    regime_id: str,
    labeled: pd.DataFrame,
    *,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_target: str,
    weights: np.ndarray,
) -> dict:
    X, y, _ = prepare_features(labeled, train_target=train_target)
    pipe, _ = train_rank_mlp(
        X[train_idx],
        y[train_idx],
        target=train_target,
        holdout_fraction=0.15,
        seed=SEED,
        hidden_layer_sizes=(256, 128, 64),
        max_iter=800,
        architecture="mask_mlp",
        source_filter="perturb_plus_search",
        target_split_ratio=TARGET_SPLIT,
        sample_weight=weights[train_idx],
        mse_weight=0.15,
    )
    split_val = labeled["split_ratio_upper"].astype(float).to_numpy()[val_idx]
    metrics = eval_val(pipe, X[val_idx], split_val, train_target=train_target)
    return {
        "regime": regime_id,
        "n_train": int(len(train_idx)),
        **metrics,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--source-filter", default="perturb_plus_search")
    args = p.parse_args()

    cfg = yaml.safe_load((REPO / args.config).read_text() if not args.config.is_absolute() else args.config.read_text())
    manifest = REPO / cfg["data"]["manifest"]
    sim_path = resolve_corpus(cfg)

    labeled = build_labeled_table(
        REPO, manifest, sim_path, recipe_version="phase0_v1", source_filter=args.source_filter
    )
    labeled = drop_invalid_targets(labeled, "split_ratio_upper")
    labeled = filter_existing_latents(labeled)
    labeled, train_target = apply_training_target(
        labeled, "abs_split_error", target_split_ratio=TARGET_SPLIT
    )
    labeled = drop_invalid_targets(labeled, train_target)

    groups = np.array(
        [extract_group(str(r["sample_id"]), str(r["source"])) for _, r in labeled.iterrows()]
    )
    sources = labeled["source"].astype(str).to_numpy()
    _, _, weights = prepare_features(labeled, train_target=train_target)

    idx = np.arange(len(labeled))
    gss = GroupShuffleSplit(n_splits=1, test_size=HOLDOUT, random_state=SEED)

    train_r, val_r = next(gss.split(idx, groups=idx))
    train_g, val_g = next(gss.split(idx, groups=groups))

    chrono_val = np.array([s in VAL_SOURCES for s in sources])
    chrono_train = np.array([s not in TRAIN_EXCLUDE_SOURCES and bool(s.strip()) for s in sources])
    if chrono_val.sum() < 5 or chrono_train.sum() < 50:
        print(
            f"warning: chronological split n_train={chrono_train.sum()} n_val={chrono_val.sum()}"
        )

    regimes = [
        ("random_holdout", train_r, val_r),
        ("sigma_group_holdout", train_g, val_g),
        ("chronological", np.where(chrono_train)[0], np.where(chrono_val)[0]),
    ]

    results = []
    for rid, tr, va in regimes:
        if len(tr) < 10 or len(va) < 3:
            results.append({"regime": rid, "status": "skipped", "n_train": len(tr), "n_val": len(va)})
            print(f"{rid}: skipped (n_train={len(tr)}, n_val={len(va)})")
            continue
        row = run_regime(rid, labeled, train_idx=tr, val_idx=va, train_target=train_target, weights=weights)
        results.append(row)
        r2 = row.get("val_r2_abs_split_error")
        r2s = f"{r2:.3f}" if r2 is not None else "—"
        print(
            f"{rid}: n_train={row['n_train']} n_val={row['n_val']} "
            f"R²={r2s} ρ={row['val_spearman_abs_err']:.3f} "
            f"top20={row['mean_abs_err_surrogate_topk']:.4f} vs rand={row['mean_abs_err_random_topk']:.4f}"
        )

    payload = {
        "corpus": str(sim_path.relative_to(REPO)),
        "source_filter": args.source_filter,
        "train_target": train_target,
        "val_sources_chronological": sorted(VAL_SOURCES),
        "train_exclude_chronological": sorted(TRAIN_EXCLUDE_SOURCES),
        "regimes": results,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "surrogate_validation.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Surrogate validation regimes",
        "",
        f"Corpus: `{payload['corpus']}` · target: `{train_target}` · rank MLP (256-128-64)",
        "",
        "| Regime | n_train | n_val | R² (|err|) | Spearman | top-20 |err| | random top-20 | wins |",
        "|--------|---------|-------|------------|----------|----------------|---------------|------|",
    ]
    for r in results:
        if r.get("status") == "skipped":
            lines.append(f"| {r['regime']} | {r['n_train']} | {r['n_val']} | — | — | — | — | skip |")
            continue
        r2 = r.get("val_r2_abs_split_error")
        r2s = f"{r2:.3f}" if r2 is not None else "—"
        win = "yes" if r.get("ranking_wins") else "no"
        lines.append(
            f"| {r['regime']} | {r['n_train']} | {r['n_val']} | {r2s} | "
            f"{r['val_spearman_abs_err']:.3f} | {r['mean_abs_err_surrogate_topk']:.4f} | "
            f"{r['mean_abs_err_random_topk']:.4f} | {win} |"
        )
    lines.extend(
        [
            "",
            "## Chronological split",
            "",
            f"Train excludes: {', '.join(sorted(TRAIN_EXCLUDE_SOURCES))}",
            f"Val sources: {', '.join(sorted(VAL_SOURCES))}",
        ]
    )
    md = OUT / "surrogate_validation.md"
    md.write_text("\n".join(lines) + "\n")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
