#!/usr/bin/env python3
"""
B5 — Surrogate acquisition: propose candidates, score by predicted FOM + exploration.

Exports top-k for MEEP verify (feeds surrogate_ranked_al_round or run_fdtd_batch).

  python scripts/surrogate_acquisition_search.py \
    --surrogate data/phase1/track_b/surrogates/perturb_latent_mlp \
    --n-proposals 1000 --top-k 30
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
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/phase1_track_b.yaml")
    p.add_argument("--surrogate", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/phase1/track_b/acquisition")
    p.add_argument("--n-proposals", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--exploration", type=float, default=0.1, help="Weight on latent distance from ref")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def acquisition_score(
    pred: float,
    target: float,
    z_flat: np.ndarray,
    ref_flat: np.ndarray,
    *,
    exploration: float,
) -> float:
    exploit = abs(pred - target)
    explore = exploration * float(np.linalg.norm(z_flat - ref_flat))
    return exploit - 0.05 * explore  # lower is better


def main() -> None:
    args = parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    target = float((cfg.get("targets") or {}).get("split_ratio_1550", 0.5))

    sur_path = args.surrogate if args.surrogate.is_absolute() else REPO_ROOT / args.surrogate
    artifact = load_artifact(sur_path / "surrogate.joblib")
    manifold = EBeamManifold.load()

    out_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    latent_dir = out_dir / "candidates/latents"
    mask_dir = out_dir / "candidates/masks"
    latent_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    ref = np.load(REPO_ROOT / "data/phase0/latents/ref_published_latent.npy").astype(np.float32)
    ref_flat = pad_latent_to_standard(ref).ravel()

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    for i in tqdm(range(args.n_proposals), desc="acquire"):
        sigma = float(rng.uniform(0.008, 0.04))
        z = pad_latent_to_standard(sample_latent_perturbation(ref, rng, sigma=sigma))
        mask = manifold.decode_numpy(z)
        if not check_mask_heuristic(mask).passed:
            continue
        pred = artifact.predict_from_latent(z, manifold=manifold)
        z_flat = z.ravel()
        score = acquisition_score(pred, target, z_flat, ref_flat, exploration=args.exploration)
        sid = f"acq_{i:05d}"
        latent_path = latent_dir / f"{sid}_latent.npy"
        mask_path = mask_dir / f"{sid}_mask.npy"
        np.save(latent_path, z)
        np.save(mask_path, mask)
        rows.append(
            {
                "sample_id": sid,
                "sigma": sigma,
                "pred_split_ratio_upper": pred,
                "acquisition_score": score,
                "latent_path": str(latent_path.relative_to(REPO_ROOT)),
                "mask_path": str(mask_path.relative_to(REPO_ROOT)),
                "drc_heuristic_pass": True,
            }
        )

    df = pd.DataFrame(rows).sort_values("acquisition_score").reset_index(drop=True)
    df.to_csv(out_dir / "acquisition_all.csv", index=False)
    top = df.head(args.top_k).copy()
    top["rank"] = np.arange(1, len(top) + 1)
    top.to_csv(out_dir / "top_candidates.csv", index=False)

    summary = {
        "n_proposals": args.n_proposals,
        "n_drc_pass": len(df),
        "top_k": args.top_k,
        "best_pred": float(top.iloc[0]["pred_split_ratio_upper"]) if len(top) else None,
        "surrogate": str(sur_path.relative_to(REPO_ROOT)),
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
    }
    (out_dir / "acquisition_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
