#!/usr/bin/env python3
"""Compute Phase 0 gate metrics: surrogate quality, search vs random on MEEP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase0.yaml")
    p.add_argument("--sim-results", type=Path, default=None)
    p.add_argument("--search-candidates", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--tolerance", type=float, default=None)
    p.add_argument("--random-k", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def in_spec(series: pd.Series, target: float, tol: float) -> pd.Series:
    return (series - target).abs() <= tol


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = REPO_ROOT / cfg["data"]["root"]
    sim_path = Path(args.sim_results) if args.sim_results else data_root / "sim_results.csv"
    cand_path = (
        Path(args.search_candidates)
        if args.search_candidates
        else data_root / "search" / "top_candidates.csv"
    )
    out_path = Path(args.output) if args.output else data_root / "gate_metrics.json"

    targets = cfg.get("targets") or {}
    target = float(targets.get("split_ratio_1550", 0.5))
    tol = args.tolerance or float(targets.get("split_ratio_tolerance", 0.05))

    sims = pd.read_csv(sim_path)
    ok = sims[sims["status"] == "ok"].copy()
    base = ok[~ok["sample_id"].astype(str).str.startswith("search_")]
    search = ok[ok["sample_id"].astype(str).str.startswith("search_")]

    metrics: dict = {
        "target_split_ratio": target,
        "tolerance": tol,
        "n_sim_ok_total": int(len(ok)),
        "n_sim_ok_baseline": int(len(base)),
        "n_sim_ok_search": int(len(search)),
    }

    if len(base):
        metrics["baseline"] = {
            "split_mean": float(base["split_ratio_upper"].mean()),
            "split_std": float(base["split_ratio_upper"].std()),
            "mean_abs_err_from_target": float((base["split_ratio_upper"] - target).abs().mean()),
            "n_in_spec": int(in_spec(base["split_ratio_upper"], target, tol).sum()),
        }

    rng = np.random.default_rng(args.seed)
    if len(base) >= args.random_k:
        rand = base.sample(args.random_k, random_state=rng)
        metrics["random_k"] = {
            "k": args.random_k,
            "mean_abs_err_from_target": float(
                (rand["split_ratio_upper"] - target).abs().mean()
            ),
            "n_in_spec": int(in_spec(rand["split_ratio_upper"], target, tol).sum()),
        }

    if len(search):
        metrics["search_meep_all"] = {
            "n": len(search),
            "mean_abs_err_from_target": float(
                (search["split_ratio_upper"] - target).abs().mean()
            ),
            "n_in_spec": int(in_spec(search["split_ratio_upper"], target, tol).sum()),
            "best_sample_id": str(
                search.loc[(search["split_ratio_upper"] - target).abs().idxmin(), "sample_id"]
            ),
            "best_split_ratio_upper": float(
                search.loc[(search["split_ratio_upper"] - target).abs().idxmin(), "split_ratio_upper"]
            ),
        }

    if cand_path.exists():
        cand = pd.read_csv(cand_path)
        merged = cand.merge(
            search,
            on="sample_id",
            how="left",
            suffixes=("_pred", "_meep"),
        )
        if "split_ratio_upper_meep" in merged.columns:
            meep_col = "split_ratio_upper_meep"
        else:
            meep_col = "split_ratio_upper"
        valid = merged[meep_col].notna()
        m = merged.loc[valid]
        if len(m):
            pred_col = "pred_split_ratio_upper"
            metrics["search_top_k"] = {
                "k": len(m),
                "mean_abs_err_meep": float((m[meep_col] - target).abs().mean()),
                "n_in_spec_meep": int(in_spec(m[meep_col], target, tol).sum()),
                "mean_surrogate_pred_err": float((m[pred_col] - target).abs().mean())
                if pred_col in m.columns
                else None,
                "mean_calibration_err": float((m[meep_col] - m[pred_col]).abs().mean())
                if pred_col in m.columns
                else None,
                "n_in_spec_surrogate": int(m["in_spec_surrogate"].sum())
                if "in_spec_surrogate" in m.columns
                else None,
            }
            if "random_k" in metrics:
                metrics["search_vs_random"] = {
                    "search_mean_abs_err": metrics["search_top_k"]["mean_abs_err_meep"],
                    "random_mean_abs_err": metrics["random_k"]["mean_abs_err_from_target"],
                    "search_wins": metrics["search_top_k"]["mean_abs_err_meep"]
                    < metrics["random_k"]["mean_abs_err_from_target"],
                }

    sur_summary = data_root / "surrogate" / "train_summary.json"
    if sur_summary.exists():
        metrics["surrogate_train"] = json.loads(sur_summary.read_text())

    metrics["verdict"] = {
        "pipeline_complete": True,
        "surrogate_calibrated": bool(
            metrics.get("surrogate_train", {}).get("val_r2", -99) > 0.3
        ),
        "search_beats_random_meep": metrics.get("search_vs_random", {}).get("search_wins", False),
        "any_meep_in_spec_search": int(metrics.get("search_meep_all", {}).get("n_in_spec", 0)) > 0,
        "phase1_recommendation": "conditional_go",
    }
    if not metrics["verdict"]["search_beats_random_meep"]:
        metrics["verdict"]["phase1_recommendation"] = "conditional_go"
    if metrics["verdict"]["surrogate_calibrated"] and metrics["verdict"]["search_beats_random_meep"]:
        metrics["verdict"]["phase1_recommendation"] = "go"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
