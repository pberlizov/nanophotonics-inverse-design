#!/usr/bin/env python3
"""Train regression vs rank-loss surrogates and compare ranking metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv/bin/python"


def run_train(out_dir: str, loss_mode: str) -> None:
    cmd = [
        str(PY),
        "scripts/train_wedge_a_surrogate.py",
        "--config",
        "configs/wedge_a_improved.yaml",
        "--skip-ranking",
    ]
    # Override output + loss via train_surrogate directly for clarity
    cmd = [
        str(PY),
        "scripts/train_surrogate.py",
        "--config",
        "configs/phase0.yaml",
        "--sim-results",
        "data/phase0/sim_results_phase0_v1_all.csv",
        "--manifest",
        "data/phase0/manifest.csv",
        "--architecture",
        "mask_mlp",
        "--sources",
        "perturb_plus_search",
        "--output-dir",
        out_dir,
        "--min-ok",
        "80",
        "--target",
        "split_ratio_upper",
        "--target-split-ratio",
        "0.5",
        "--hidden",
        "256,128,64",
        "--max-iter",
        "400",
        "--mask-pool",
        "6",
        "--sigma-feature",
        "--decode-masks-from-latent",
        "--sample-weight-mode",
        "in_spec_boost",
        "--sample-weight-in-spec-tol",
        "0.05",
        "--champion-weight",
        "2.0",
        "--loss-mode",
        loss_mode,
        "--champion-latent",
        "data/phase0/latents/ref_published_latent.npy",
        "--champion-latent",
        "data/phase1/meep_search_local/candidates/latents/local_00022_latent.npy",
        "--champion-latent",
        "data/phase1/meep_search_deep/candidates/latents/meep_bo_00128_latent.npy",
        "--champion-latent",
        "data/phase1/wedge_a/meep_gated_shortlist/latents/cand_000160_latent.npy",
        "--champion-latent",
        "data/phase1/wedge_a/meep_gated_shortlist/round3/latents/cand_000003_latent.npy",
    ]
    if loss_mode == "pairwise_rank":
        cmd.extend(["--rank-mse-weight", "0.15"])
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)


def eval_ranking(surrogate_dir: str, out_json: str) -> dict:
    cmd = [
        str(PY),
        "scripts/evaluate_surrogate_ranking.py",
        "--surrogate",
        surrogate_dir,
        "--sim-results",
        "data/phase0/sim_results_phase0_v1_all.csv",
        "--sources",
        "perturb_plus_search",
        "--top-k",
        "20",
        "--target",
        "0.5",
        "--output",
        out_json,
    ]
    subprocess.run(cmd, cwd=REPO, check=True)
    return json.loads((REPO / out_json).read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    base = REPO / "data/phase1/wedge_a"
    reg_dir = "data/phase1/wedge_a/surrogate_improved_baseline_cmp"
    rank_dir = "data/phase1/wedge_a/surrogate_improved_rank_cmp"

    if not args.skip_train:
        run_train(reg_dir, "regression")
        run_train(rank_dir, "pairwise_rank")

    reg = eval_ranking(reg_dir, "data/phase1/wedge_a/ranking_cmp_regression.json")
    rank = eval_ranking(rank_dir, "data/phase1/wedge_a/ranking_cmp_pairwise.json")

    summary = {
        "regression": {
            "spearman_err": reg.get("spearman_err"),
            "ranking_wins": reg.get("ranking_wins"),
            "mean_abs_err_topk": reg.get("mean_abs_err_surrogate_topk"),
            "n_in_spec_topk": reg.get("n_in_spec_surrogate_topk"),
        },
        "pairwise_rank": {
            "spearman_err": rank.get("spearman_err"),
            "ranking_wins": rank.get("ranking_wins"),
            "mean_abs_err_topk": rank.get("mean_abs_err_surrogate_topk"),
            "n_in_spec_topk": rank.get("n_in_spec_surrogate_topk"),
        },
    }
    out = base / "rank_loss_comparison.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
