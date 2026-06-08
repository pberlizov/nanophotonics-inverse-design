#!/usr/bin/env python3
"""Compare split sensitivity across MEEP recipes on champion masks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.crosscheck.types import SolverSpec
from nano_inv.crosscheck.meep_backend import run_meep

ALL_CHAMPIONS = [
    ("local_00022", "data/phase1/meep_search_local/candidates/masks/local_00022_mask.npy"),
    ("meep_bo_00128", "data/phase1/meep_search_deep/candidates/masks/meep_bo_00128_mask.npy"),
    ("meep_bo_00093", "data/phase0/meep_search_100/candidates/masks/meep_bo_00093_mask.npy"),
]

# Focused sweep for contract promotion (see docs/RECIPE_SENSITIVITY.md).
SPECS = [
    SolverSpec("meep_v1_r25", "meep", "phase0_v1", 25, "production"),
    SolverSpec("meep_v1_r50", "meep", "phase0_v1", 50, "production"),
    SolverSpec("meep_v1_matgrid_r25", "meep", "phase0_v1_matgrid", 25, "MaterialGrid"),
    SolverSpec("meep_v1_matgrid_r50", "meep", "phase0_v1_matgrid", 50, "MaterialGrid"),
    SolverSpec("meep_v1_matgrid_avg_r25", "meep", "phase0_v1_matgrid_avg", 25, "MaterialGrid+avg"),
    SolverSpec("meep_v1_matgrid_avg_r50", "meep", "phase0_v1_matgrid_avg", 50, "MaterialGrid+avg"),
    SolverSpec("meep_v1_refgrid100n_r25", "meep", "phase0_v1_refgrid100n", 25, "refgrid100n"),
    SolverSpec("meep_v1_refgrid100n_r50", "meep", "phase0_v1_refgrid100n", 50, "refgrid100n"),
    SolverSpec("meep_v1_refgrid25n_r25", "meep", "phase0_v1_refgrid25n", 25, "refgrid25n"),
    SolverSpec("meep_v1_refgrid25n_r50", "meep", "phase0_v1_refgrid25n", 50, "refgrid25n"),
]

OUT = REPO_ROOT / "data/phase1/recipe_sensitivity"


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--sample-id", default=None, help="Run one mask only (e.g. local_00022)")
    args = p.parse_args()

    champions = ALL_CHAMPIONS
    if args.sample_id:
        champions = [c for c in ALL_CHAMPIONS if c[0] == args.sample_id]
        if not champions:
            raise SystemExit(f"unknown sample_id {args.sample_id!r}")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    ref_splits: dict[str, float] = {}

    for sid, rel in champions:
        mask = np.load(REPO_ROOT / rel)
        for spec in SPECS:
            print(f"==> {sid} / {spec.name}")
            r = run_meep(sid, rel, mask, spec)
            row = r.to_dict()
            rows.append(row)
            if spec.name == "meep_v1_r25" and r.status == "ok":
                ref_splits[sid] = r.split_ratio_upper
            (OUT / "sensitivity.json").write_text(json.dumps(rows, indent=2))

    for row in rows:
        sid = row["sample_id"]
        ref = ref_splits.get(sid)
        if ref is not None and row["status"] == "ok":
            row["delta_vs_v1_r25"] = abs(float(row["split_ratio_upper"]) - ref)

    (OUT / "sensitivity.json").write_text(json.dumps(rows, indent=2))

    lines = [
        "# Recipe sensitivity study",
        "",
        "Reference: `meep_v1_r25` (production). Goal: r50 within ~0.03 of r25 on same mask.",
        "",
        "| sample | solver | split | Δ vs r25 |",
        "|--------|--------|-------|----------|",
    ]
    for row in rows:
        d = row.get("delta_vs_v1_r25")
        ds = f"{d:.3f}" if d is not None else "—"
        sp = row.get("split_ratio_upper", float("nan"))
        lines.append(
            f"| {row['sample_id']} | {row['solver']} | {sp:.3f} | {ds} |"
        )

    # Summary: max Δ vs r25 per solver (excluding r25 itself).
    by_solver: dict[str, list[float]] = {}
    for row in rows:
        d = row.get("delta_vs_v1_r25")
        if d is not None and "r25" not in row["solver"]:
            by_solver.setdefault(row["solver"], []).append(float(d))
    lines.append("\n## Max Δ vs v1_r25 (per solver, all champions)\n")
    for name, deltas in sorted(by_solver.items(), key=lambda x: max(x[1])):
        lines.append(f"- `{name}`: max **{max(deltas):.3f}**, mean {np.mean(deltas):.3f}")

    lines.append("\n## r25↔r50 gap (same mask, |split_r50 − split_r25|)\n")
    lines.append("| sample | recipe | gap |")
    lines.append("|--------|--------|-----|")
    sample_ids = sorted({r["sample_id"] for r in rows})
    pairs = [
        ("meep_v1", "production"),
        ("meep_v1_matgrid", "matgrid"),
        ("meep_v1_matgrid_avg", "matgrid_avg"),
        ("meep_v1_refgrid100n", "refgrid100n"),
        ("meep_v1_refgrid25n", "refgrid25n"),
    ]
    for sid in sample_ids:
        for prefix, label in pairs:
            r25_row = next(
                (r for r in rows if r["sample_id"] == sid and r["solver"] == f"{prefix}_r25"),
                None,
            )
            r50_row = next(
                (r for r in rows if r["sample_id"] == sid and r["solver"] == f"{prefix}_r50"),
                None,
            )
            if not r25_row or not r50_row or r25_row["status"] != "ok" or r50_row["status"] != "ok":
                continue
            gap = abs(float(r50_row["split_ratio_upper"]) - float(r25_row["split_ratio_upper"]))
            lines.append(f"| {sid} | {label} | {gap:.3f} |")

    (OUT / "sensitivity_report.md").write_text("\n".join(lines))
    print(f"Wrote {OUT / 'sensitivity_report.md'}")


if __name__ == "__main__":
    main()
