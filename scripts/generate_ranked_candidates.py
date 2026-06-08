#!/usr/bin/env python3
"""Generate DRC-pass candidates + surrogate scores (.venv only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def as_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())

from nano_inv.champions import load_champion_centers  # noqa: E402
from nano_inv.drc_heuristic import check_mask_heuristic  # noqa: E402
from nano_inv.latent import pad_latent_to_standard, sample_latent_perturbation  # noqa: E402
from nano_inv.manifold import EBeamManifold  # noqa: E402
from nano_inv.pilot import load_pilot_config  # noqa: E402
from nano_inv.surrogate import load_artifact, surrogate_ranking_scores  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/wedge_a.yaml")
    p.add_argument("--surrogate", type=Path, default=None)
    p.add_argument("--n-proposals", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sigma-min", type=float, default=0.008)
    p.add_argument("--sigma-max", type=float, default=0.04)
    p.add_argument("--center-sigma", type=float, default=None)
    p.add_argument("--sigma-span", type=float, default=None, help="Half-width if center-sigma set")
    p.add_argument(
        "--latent-only",
        action="store_true",
        help="Save latents only (decode mask at MEEP time; ~50%% less disk)",
    )
    args = p.parse_args()

    cfg = load_pilot_config(args.config if args.config.is_absolute() else REPO_ROOT / args.config)
    target = float(cfg["targets"]["split_ratio_1550"])
    sur_cfg = cfg.get("surrogate") or {}
    sur_path = args.surrogate or REPO_ROOT / sur_cfg["output_dir"]
    if not sur_path.is_absolute():
        sur_path = REPO_ROOT / sur_path

    artifact = load_artifact(sur_path / "surrogate.joblib")
    manifold = EBeamManifold.load()
    ref = np.load(REPO_ROOT / "data/phase0/latents/ref_published_latent.npy").astype(np.float32)
    champ_cfg = cfg.get("champions", {})
    centers = load_champion_centers(
        REPO_ROOT, champ_cfg.get("latent_paths") if champ_cfg.get("enabled", True) else None
    )
    rng = np.random.default_rng(args.seed)

    rows: list[dict] = []
    for i in tqdm(range(args.n_proposals), desc="proposals"):
        if args.center_sigma is not None:
            span = args.sigma_span or 0.01
            sigma = float(rng.uniform(max(0.005, args.center_sigma - span), args.center_sigma + span))
        else:
            sigma = float(rng.uniform(args.sigma_min, args.sigma_max))
        center = centers[int(rng.integers(0, len(centers)))] if centers else ref
        z = pad_latent_to_standard(sample_latent_perturbation(center, rng, sigma=sigma))
        mask = manifold.decode_numpy(z)
        if not check_mask_heuristic(mask).passed:
            continue
        pred = float(artifact.predict_from_latent(z, manifold=manifold, sigma=sigma))
        score = float(
            surrogate_ranking_scores(
                np.array([pred]),
                target=artifact.target,
                target_split_ratio=float(getattr(artifact, "target_split_ratio", target)),
            )[0]
        )
        sid = f"cand_{i:06d}"
        out_base = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        out_base = out_base.resolve().parent
        latent_dir = out_base / "latents"
        mask_dir = out_base / "masks"
        latent_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        lp = (latent_dir / f"{sid}_latent.npy").resolve()
        mp = (mask_dir / f"{sid}_mask.npy").resolve()
        np.save(lp, z)
        if not args.latent_only:
            np.save(mp, mask)
        rows.append(
            {
                "sample_id": sid,
                "sigma": sigma,
                "pred_split_ratio_upper": pred,
                "surrogate_score": score,
                "latent_path": as_repo_relative(lp),
                "mask_path": as_repo_relative(mp) if not args.latent_only else "",
            }
        )

    df = pd.DataFrame(rows).sort_values("surrogate_score").reset_index(drop=True)
    out = (REPO_ROOT / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  n={len(df)}")


if __name__ == "__main__":
    main()
