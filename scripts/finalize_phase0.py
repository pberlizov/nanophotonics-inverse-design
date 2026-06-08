#!/usr/bin/env python3
"""Merge v1 labels + meep_search designs, train final surrogate, write closeout JSON."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.manifest import build_al_training_manifest, filter_by_source  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase0.yaml")
    p.add_argument("--best-id", type=str, default="meep_bo_00010")
    p.add_argument("--skip-train", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    data_root = REPO_ROOT / cfg["data"]["root"]

    v1_path = data_root / "sim_results_phase0_v1.csv"
    main_sim = data_root / "sim_results.csv"
    meep_top = data_root / "meep_search" / "top_candidates.csv"
    out_sim = data_root / "sim_results_phase0_final.csv"
    out_manifest = data_root / "manifest_phase0_final.csv"
    out_sur = data_root / "surrogate_phase0_final"

    if not v1_path.exists():
        raise SystemExit(f"missing {v1_path} — run relabel_recipe.py first")
    v1 = pd.read_csv(v1_path)
    extra = pd.DataFrame()
    if main_sim.exists():
        all_sim = pd.read_csv(main_sim)
        extra = all_sim[all_sim["sample_id"].astype(str).str.startswith("meep_bo")]
    combined = pd.concat([v1, extra], ignore_index=True)
    combined = combined.drop_duplicates(subset=["sample_id"], keep="last")
    combined.to_csv(out_sim, index=False)

    base = filter_by_source(pd.read_csv(REPO_ROOT / cfg["data"]["manifest"]), "perturb")
    extras: list[pd.DataFrame] = []
    for cand_path in (
        meep_top,
        data_root / "meep_search_100" / "top_candidates.csv",
    ):
        if not cand_path.exists():
            continue
        c = pd.read_csv(cand_path)
        rows = []
        for _, r in c.iterrows():
            rows.append(
                {
                    "sample_id": r["sample_id"],
                    "source": "meep_native",
                    "latent_path": r["latent_path"],
                    "mask_path": r["mask_path"],
                    "mask_shape_h": 180,
                    "mask_shape_w": 180,
                    "drc_heuristic_pass": True,
                }
            )
        extras.append(pd.DataFrame(rows))
    m = base if not extras else pd.concat([base, *extras], ignore_index=True)
    m = m.drop_duplicates(subset=["sample_id"], keep="last")
    m.to_csv(out_manifest, index=False)

    summary = {
        "sim_results_final": str(out_sim.relative_to(REPO_ROOT)),
        "manifest_final": str(out_manifest.relative_to(REPO_ROOT)),
        "n_sim_rows": len(combined),
        "n_meep_native": int(
            combined["sample_id"].astype(str).str.startswith("meep_bo").sum()
        ),
        "n_manifest_rows": len(pd.read_csv(out_manifest)),
        "best_design_id": args.best_id,
    }
    if args.best_id in set(combined["sample_id"].astype(str)):
        row = combined[combined["sample_id"] == args.best_id].iloc[-1]
        summary["best_design"] = {
            "split_ratio_upper": float(row["split_ratio_upper"]),
            "recipe_version": str(row.get("recipe_version", "")),
            "sigma": float(
                pd.read_csv(meep_top).loc[
                    pd.read_csv(meep_top)["sample_id"] == args.best_id, "sigma"
                ].iloc[0]
            )
            if meep_top.exists()
            else None,
        }

    if not args.skip_train:
        py = REPO_ROOT / ".venv/bin/python"
        subprocess.run(
            [
                str(py),
                str(REPO_ROOT / "scripts/train_surrogate.py"),
                "--manifest",
                str(out_manifest.relative_to(REPO_ROOT)),
                "--sim-results",
                str(out_sim.relative_to(REPO_ROOT)),
                "--architecture",
                "mask_mlp",
                "--sources",
                "all",
                "--output-dir",
                str(out_sur.relative_to(REPO_ROOT)),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        ts = json.loads((out_sur / "train_summary.json").read_text())
        summary["surrogate_train"] = ts

    closeout_path = data_root / "phase0_closeout.json"
    closeout_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
