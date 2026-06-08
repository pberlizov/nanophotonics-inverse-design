#!/usr/bin/env python3
"""
Cheap surrogate pre-search: decode many perturbations, rank by |pred_split - target|.

MEEP verifies only the top-k (see surrogate_ranked_al_round.py).

  python scripts/surrogate_ranked_presearch.py \
    --surrogate data/phase1/surrogate_mask_v1_full \
    --n-proposals 800 --top-k 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.drc_heuristic import check_mask_heuristic  # noqa: E402
from nano_inv.latent import pad_latent_to_standard, sample_latent_perturbation  # noqa: E402
from nano_inv.manifold import EBeamManifold  # noqa: E402
from nano_inv.surrogate import load_artifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase1.yaml")
    p.add_argument("--surrogate", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/phase1/surrogate_presearch")
    p.add_argument("--n-proposals", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sigma-min", type=float, default=None)
    p.add_argument("--sigma-max", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    pre_cfg = cfg.get("surrogate_presearch") or {}
    targets = cfg.get("targets") or {}
    target = float(targets.get("split_ratio_1550", 0.5))

    n_prop = args.n_proposals or int(pre_cfg.get("n_proposals", 500))
    top_k = args.top_k or int(pre_cfg.get("top_k", 30))
    s_min = args.sigma_min if args.sigma_min is not None else float(pre_cfg.get("sigma_min", 0.008))
    s_max = args.sigma_max if args.sigma_max is not None else float(pre_cfg.get("sigma_max", 0.04))

    sur_path = args.surrogate
    if not sur_path.is_absolute():
        sur_path = REPO_ROOT / sur_path
    artifact = load_artifact(sur_path / "surrogate.joblib")

    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    latent_dir = out_dir / "candidates" / "latents"
    mask_dir = out_dir / "candidates" / "masks"
    latent_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    ref_path = REPO_ROOT / "data/phase0/latents/ref_published_latent.npy"
    ref = np.load(ref_path).astype(np.float32)
    manifold = EBeamManifold.load()
    rng = np.random.default_rng(args.seed)

    rows: list[dict] = []
    for i in tqdm(range(n_prop), desc="presearch"):
        sigma = float(rng.uniform(s_min, s_max))
        z = sample_latent_perturbation(ref, rng, sigma=sigma)
        z_std = pad_latent_to_standard(z)
        sid = f"pre_{i:05d}"
        latent_path = latent_dir / f"{sid}_latent.npy"
        mask_path = mask_dir / f"{sid}_mask.npy"
        np.save(latent_path, z_std)
        mask = manifold.decode_numpy(z_std)
        np.save(mask_path, mask)
        drc = check_mask_heuristic(mask)
        if not drc.passed:
            continue
        pred = artifact.predict_mask(mask)
        score = abs(pred - target)
        rows.append(
            {
                "sample_id": sid,
                "sigma": sigma,
                "pred_split_ratio_upper": pred,
                "surrogate_score": score,
                "latent_path": str(latent_path.relative_to(REPO_ROOT)),
                "mask_path": str(mask_path.relative_to(REPO_ROOT)),
                "drc_heuristic_pass": drc.passed,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no DRC-passing proposals")
    df = df.sort_values("surrogate_score").reset_index(drop=True)
    df.to_csv(out_dir / "presearch_all.csv", index=False)

    top = df.head(top_k).copy()
    top["rank"] = np.arange(1, len(top) + 1)
    top.to_csv(out_dir / "top_candidates.csv", index=False)

    summary = {
        "n_proposals": n_prop,
        "n_drc_pass": len(df),
        "top_k": top_k,
        "best_pred_split": float(top.iloc[0]["pred_split_ratio_upper"]),
        "best_surrogate_score": float(top.iloc[0]["surrogate_score"]),
        "sigma_range": [s_min, s_max],
        "surrogate": str(sur_path.relative_to(REPO_ROOT)),
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
    }
    (out_dir / "presearch_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
