#!/usr/bin/env python3
"""Export PNG (+ metadata) for best MEEP design."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def as_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-id", type=str, default="meep_bo_00093")
    p.add_argument("--mask", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/phase0/exports")
    args = p.parse_args()

    if args.mask:
        mask_path = resolve_repo_path(args.mask)
    else:
        for base in (
            REPO_ROOT / "data/phase1/meep_search_local/candidates/masks",
            REPO_ROOT / "data/phase1/meep_search/candidates/masks",
            REPO_ROOT / "data/phase0/meep_search_100/candidates/masks",
            REPO_ROOT / "data/phase0/meep_search/candidates/masks",
        ):
            candidate = base / f"{args.sample_id}_mask.npy"
            if candidate.exists():
                mask_path = candidate
                break
        else:
            mask_path = (
                REPO_ROOT / "data/phase0/meep_search_100/candidates/masks" / f"{args.sample_id}_mask.npy"
            )
    if not mask_path.exists():
        raise SystemExit(f"mask not found: {mask_path}")

    mask = np.load(mask_path)
    out_dir = resolve_repo_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {"sample_id": args.sample_id, "mask_path": as_repo_relative(mask_path)}
    import pandas as pd

    for sim_path in (
        REPO_ROOT / "data/phase1/meep_search_local/meep_search_local_trials.csv",
        REPO_ROOT / "data/phase0/sim_results_phase0_v1_all.csv",
        REPO_ROOT / "data/phase0/sim_results.csv",
    ):
        if not sim_path.exists():
            continue
        s = pd.read_csv(sim_path)
        key = "sample_id" if "sample_id" in s.columns else None
        if key is None:
            continue
        rows = s[s[key].astype(str) == args.sample_id]
        if len(rows):
            r = rows.iloc[-1]
            split_col = "split_ratio_upper" if "split_ratio_upper" in r else "meep_split_ratio_upper"
            if split_col in r and pd.notna(r[split_col]):
                meta["split_ratio_upper"] = float(r[split_col])
            if "recipe_version" in r and pd.notna(r.get("recipe_version")):
                meta["recipe_version"] = str(r["recipe_version"])
            if "insertion_loss_db" in r and pd.notna(r["insertion_loss_db"]):
                meta["insertion_loss_db"] = float(r["insertion_loss_db"])
            break

    (out_dir / f"{args.sample_id}_meta.json").write_text(json.dumps(meta, indent=2))

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(5, 5))
        plt.imshow(mask, cmap="gray", interpolation="nearest")
        plt.title(f"{args.sample_id}\nsplit={meta.get('split_ratio_upper', '?')}")
        plt.axis("off")
        plt.tight_layout()
        png = out_dir / f"{args.sample_id}_mask.png"
        plt.savefig(png, dpi=200)
        plt.close()
        print(f"wrote {png}")
    except ImportError:
        print("matplotlib not installed; wrote metadata only")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
