#!/usr/bin/env python3
"""Batch-decode EBL latents through drcgenerator and write Phase 0 manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.drc_heuristic import check_mask_heuristic  # noqa: E402
from nano_inv.manifold import (  # noqa: E402
    EBeamManifold,
    sample_latent_perlin,
    sample_latent_perturbation,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-samples", type=int, default=500, help="Total decodes to generate")
    p.add_argument(
        "--mix",
        type=str,
        default="0.2,0.8",
        help="Fraction reference,perlin (must sum to 1)",
    )
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data" / "phase0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preview-png", action="store_true", help="Save a few PNG previews")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frac_ref, frac_perlin = (float(x) for x in args.mix.split(","))
    if abs(frac_ref + frac_perlin - 1.0) > 1e-6:
        raise SystemExit("--mix must sum to 1.0")

    n_ref = int(args.n_samples * frac_ref)
    n_perlin = args.n_samples - n_ref

    out = args.output_dir
    latent_dir = out / "latents"
    mask_dir = out / "masks"
    latent_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    manifold = EBeamManifold.load()
    ref_np = np.asarray(manifold.reference_latent, dtype=np.float32)

    rows: list[dict] = []

    def process_one(sample_id: str, latent: np.ndarray, source: str, meta: dict) -> None:
        mask = manifold.decode_numpy(latent)
        latent_path = latent_dir / f"{sample_id}_latent.npy"
        mask_path = mask_dir / f"{sample_id}_mask.npy"
        np.save(latent_path, latent)
        np.save(mask_path, mask)

        drc = check_mask_heuristic(mask)
        rows.append(
            {
                "sample_id": sample_id,
                "source": source,
                "latent_path": str(latent_path.relative_to(REPO_ROOT)),
                "mask_path": str(mask_path.relative_to(REPO_ROOT)),
                "mask_shape_h": mask.shape[0],
                "mask_shape_w": mask.shape[1],
                "fill_ratio": drc.fill_ratio,
                "min_run_length": drc.min_run_length,
                "drc_heuristic_pass": drc.passed,
                "drc_reasons": ";".join(drc.reasons),
                **meta,
            }
        )

    # Published 50/50 optimum
    process_one("ref_published", ref_np, "published_reference", {})

    # Perturbations around reference
    for i in tqdm(range(n_ref), desc="perturb"):
        sigma = float(rng.uniform(0.03, 0.15))
        z = sample_latent_perturbation(ref_np, rng, sigma=sigma)
        process_one(f"pert_{i:05d}", z, "perturbation", {"sigma": sigma})

    # Perlin latents (notebook-style)
    for i in tqdm(range(n_perlin), desc="perlin"):
        scale = float(rng.uniform(2.5, 5.0))
        offset = (float(rng.uniform(0, 30)), float(rng.uniform(0, 30)))
        dim = int(rng.choice([2, 3, 4]))
        z = sample_latent_perlin(scale=scale, offset=offset, dim=dim)
        process_one(
            f"perlin_{i:05d}",
            z,
            "perlin",
            {"scale": scale, "offset_x": offset[0], "offset_y": offset[1], "dim": dim},
        )

    manifest_path = out / "manifest.csv"
    df = pd.DataFrame(rows)
    df.to_csv(manifest_path, index=False)

    n_pass = int(df["drc_heuristic_pass"].sum())
    report = {
        "n_total": len(df),
        "n_pass_heuristic": n_pass,
        "pass_rate": n_pass / max(len(df), 1),
        "manifest": str(manifest_path.relative_to(REPO_ROOT)),
    }
    report_path = out / "drc_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))

    if args.preview_png:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skip --preview-png")
            return

        preview_dir = out / "previews"
        preview_dir.mkdir(exist_ok=True)
        for sample_id in df.head(8)["sample_id"]:
            mask = np.load(mask_dir / f"{sample_id}_mask.npy")
            plt.figure(figsize=(4, 4))
            plt.imshow(mask, cmap="gray", interpolation="nearest")
            plt.title(sample_id)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(preview_dir / f"{sample_id}.png", dpi=120)
            plt.close()


if __name__ == "__main__":
    main()
