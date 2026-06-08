#!/usr/bin/env python3
"""Option B mesh-stability sprint: HDF5 epsilon_file + fixed-runtime controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nano_inv.crosscheck.meep_backend import run_meep
from nano_inv.crosscheck.types import SolverSpec

CHAMPIONS = [
    ("local_00022", "data/phase1/meep_search_local/candidates/masks/local_00022_mask.npy"),
    ("meep_bo_00128", "data/phase1/meep_search_deep/candidates/masks/meep_bo_00128_mask.npy"),
    ("meep_bo_00093", "data/phase0/meep_search_100/candidates/masks/meep_bo_00093_mask.npy"),
]

SPECS = [
    SolverSpec("prod_r25", "meep", "phase0_v1", 25, "production"),
    SolverSpec("prod_r50", "meep", "phase0_v1", 50, "production"),
    SolverSpec("v1_fixed_r25", "meep", "phase0_v1_fixed", 25, "v1 fixed t=2520"),
    SolverSpec("v1_fixed_r50", "meep", "phase0_v1_fixed", 50, "v1 fixed t=2520"),
    SolverSpec("epsfile_r25", "meep", "phase0_v1_epsfile", 25, "HDF5 ε @100"),
    SolverSpec("epsfile_r50", "meep", "phase0_v1_epsfile", 50, "HDF5 ε @100"),
    SolverSpec("epsfile_fixed_r25", "meep", "phase0_v1_epsfile_fixed", 25, "HDF5+fixed t"),
    SolverSpec("epsfile_fixed_r50", "meep", "phase0_v1_epsfile_fixed", 50, "HDF5+fixed t"),
    SolverSpec("matgrid_r25", "meep", "phase0_v1_matgrid", 25, "MaterialGrid"),
    SolverSpec("matgrid_r50", "meep", "phase0_v1_matgrid", 50, "MaterialGrid"),
    SolverSpec("refgrid100n_r25", "meep", "phase0_v1_refgrid100n", 25, "refgrid100n"),
    SolverSpec("refgrid100n_r50", "meep", "phase0_v1_refgrid100n", 50, "refgrid100n"),
]

OUT = REPO / "data" / "phase1" / "recipe_sensitivity"
PAIRS = [
    ("prod", "prod_r25", "prod_r50"),
    ("v1_fixed", "v1_fixed_r25", "v1_fixed_r50"),
    ("epsfile", "epsfile_r25", "epsfile_r50"),
    ("epsfile_fixed", "epsfile_fixed_r25", "epsfile_fixed_r50"),
    ("matgrid", "matgrid_r25", "matgrid_r50"),
    ("refgrid100n", "refgrid100n_r25", "refgrid100n_r50"),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    prod_r25: dict[str, float] = {}

    for sid, rel in CHAMPIONS:
        mask = np.load(REPO / rel)
        for spec in SPECS:
            print(f"==> {sid} / {spec.name}")
            r = run_meep(sid, rel, mask, spec)
            row = r.to_dict()
            rows.append(row)
            if spec.name == "prod_r25" and r.status == "ok":
                prod_r25[sid] = float(r.split_ratio_upper)
            (OUT / "mesh_stable_b.json").write_text(json.dumps(rows, indent=2))

    lines = [
        "# Option B mesh-stability sprint",
        "",
        "Dual criterion: **r25↔r50 gap ≤ 0.03** and **|r25 − production r25| ≤ 0.03**.",
        "",
        "## r25↔r50 gap",
        "",
        "| sample | recipe | r25 | r50 | gap | |r25−prod| |",
        "|--------|--------|-----|-----|-----|------------|",
    ]
    by_recipe_gap: dict[str, list[float]] = {}
    for sid in sorted({r["sample_id"] for r in rows}):
        pref = prod_r25.get(sid, float("nan"))
        for label, s25, s50 in PAIRS:
            r25_row = next((r for r in rows if r["sample_id"] == sid and r["solver"] == s25), None)
            r50_row = next((r for r in rows if r["sample_id"] == sid and r["solver"] == s50), None)
            if not r25_row or not r50_row or r25_row["status"] != "ok" or r50_row["status"] != "ok":
                continue
            s25v = float(r25_row["split_ratio_upper"])
            s50v = float(r50_row["split_ratio_upper"])
            gap = abs(s50v - s25v)
            dprod = abs(s25v - pref) if np.isfinite(pref) else float("nan")
            by_recipe_gap.setdefault(label, []).append(gap)
            lines.append(
                f"| {sid} | {label} | {s25v:.3f} | {s50v:.3f} | {gap:.3f} | {dprod:.3f} |"
            )

    lines.append("\n## Max r25↔r50 gap (all champions)\n")
    for label, gaps in sorted(by_recipe_gap.items(), key=lambda x: max(x[1])):
        lines.append(f"- `{label}`: max **{max(gaps):.3f}**, mean {np.mean(gaps):.3f}")

    winners = [
        label
        for label, gaps in by_recipe_gap.items()
        if max(gaps) <= 0.03
    ]
    lines.append("\n## Recipes with max gap ≤ 0.03\n")
    lines.append(", ".join(winners) if winners else "_none yet_")

    (OUT / "mesh_stable_b_report.md").write_text("\n".join(lines))
    print(f"Wrote {OUT / 'mesh_stable_b_report.md'}")


if __name__ == "__main__":
    main()
