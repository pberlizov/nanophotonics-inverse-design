#!/usr/bin/env python3
"""Train wedge A ranker from config (supports mask_mlp, abs_split_error, custom hidden)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.pilot import load_pilot_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/wedge_a.yaml")
    p.add_argument("--skip-ranking", action="store_true")
    args = p.parse_args()

    cfg = load_pilot_config(args.config)
    sur = cfg["surrogate"]
    gate = cfg["ranking_gate"]
    targets = cfg.get("targets", {})
    py = REPO_ROOT / ".venv/bin/python"

    cmd = [
        str(py),
        "scripts/train_surrogate.py",
        "--config",
        str(REPO_ROOT / "configs/phase0.yaml"),
        "--sim-results",
        cfg["data"]["sim_corpus"],
        "--manifest",
        cfg["data"].get("manifest", "data/phase0/manifest.csv"),
        "--architecture",
        sur["architecture"],
        "--sources",
        sur["source_filter"],
        "--output-dir",
        sur["output_dir"],
        "--min-ok",
        str(sur["min_ok"]),
        "--target",
        str(sur.get("target", "split_ratio_upper")),
        "--target-split-ratio",
        str(targets.get("split_ratio_1550", 0.5)),
    ]
    if sur.get("hidden"):
        cmd.extend(["--hidden", str(sur["hidden"])])
    if sur.get("max_iter"):
        cmd.extend(["--max-iter", str(sur["max_iter"])])
    if sur.get("mask_pool"):
        cmd.extend(["--mask-pool", str(sur["mask_pool"])])
    if sur.get("near_target_max_abs_err") is not None:
        cmd.extend(["--near-target-max-abs-err", str(sur["near_target_max_abs_err"])])
    if sur.get("sigma_feature"):
        cmd.append("--sigma-feature")
    if sur.get("decode_masks_from_latent"):
        cmd.append("--decode-masks-from-latent")
    if sur.get("sample_weight_mode"):
        cmd.extend(["--sample-weight-mode", str(sur["sample_weight_mode"])])
    if sur.get("sample_weight_in_spec_tol") is not None:
        cmd.extend(
            ["--sample-weight-in-spec-tol", str(sur["sample_weight_in_spec_tol"])]
        )
    if sur.get("champion_weight") is not None:
        cmd.extend(["--champion-weight", str(sur["champion_weight"])])
    champs = cfg.get("champions") or {}
    if champs.get("enabled") and champs.get("latent_paths"):
        for lp in champs["latent_paths"]:
            cmd.extend(["--champion-latent", lp])
    if sur.get("loss_mode"):
        cmd.extend(["--loss-mode", str(sur["loss_mode"])])
    if sur.get("rank_mse_weight") is not None:
        cmd.extend(["--rank-mse-weight", str(sur["rank_mse_weight"])])

    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    if args.skip_ranking:
        return

    out_eval = REPO_ROOT / cfg["data"]["wedge_root"] / "ranking_eval.json"
    if sur.get("output_dir", "").endswith("mask_perturb"):
        out_eval = REPO_ROOT / "data/phase1/wedge_a/ranking_eval_mask_perturb.json"
    elif "abserr" in str(sur.get("output_dir", "")):
        out_eval = REPO_ROOT / "data/phase1/wedge_a/ranking_eval_mask_abserr.json"
    elif "improved" in str(sur.get("output_dir", "")):
        out_eval = REPO_ROOT / "data/phase1/wedge_a/ranking_eval_improved.json"

    eval_cmd = [
        str(py),
        "scripts/evaluate_surrogate_ranking.py",
        "--surrogate",
        sur["output_dir"],
        "--sim-results",
        cfg["data"]["sim_corpus"],
        "--sources",
        gate["sources"],
        "--top-k",
        str(gate["top_k"]),
        "--target",
        str(targets.get("split_ratio_1550", 0.5)),
        "--output",
        str(out_eval.relative_to(REPO_ROOT)),
    ]
    print("+", " ".join(eval_cmd))
    subprocess.run(eval_cmd, cwd=REPO_ROOT, check=True)

    metrics = json.loads(out_eval.read_text())
    if gate.get("require_ranking_wins") and not metrics.get("ranking_wins"):
        print("WARNING: ranking_wins is false", file=sys.stderr)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
