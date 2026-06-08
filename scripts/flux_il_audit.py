#!/usr/bin/env python3
"""Audit flux-monitor sensitivity (aperture + in-plane offset) for champions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nano_inv.meep_sim import MeepRecipe, require_meep, simulate_mask  # noqa: E402

OUT = REPO / "data/phase1/release"
DEFAULT_CFG = REPO / "configs/promote_sdf_geom.yaml"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument(
        "--flux-y-scales",
        type=str,
        default="0.6,0.8,1.0,1.2",
        help="Monitor height as fraction of wg_width_um",
    )
    p.add_argument(
        "--in-monitor-dx",
        type=str,
        default="0.05,0.1,0.2",
        help="Input flux plane offset from design edge (µm)",
    )
    args = p.parse_args()

    require_meep()
    cfg = yaml.safe_load(args.config.read_text())
    recipe = MeepRecipe.for_version(
        cfg.get("production_recipe", "phase0_v1"),
        {"resolution": 25, "mask_flip_y": True},
    )
    scales = [float(x) for x in args.flux_y_scales.split(",")]
    dxs = [float(x) for x in args.in_monitor_dx.split(",")]

    rows: list[dict] = []
    for entry in cfg.get("champions", []):
        sid = entry["id"]
        mask = np.load(REPO / entry["mask"])
        for fy in scales:
            for dx in dxs:
                res = simulate_mask(
                    mask,
                    recipe,
                    verbose=False,
                    flux_y_scale=fy,
                    in_monitor_dx=dx,
                )
                t_total = (
                    (res.flux_out_upper + res.flux_out_lower) / res.flux_in
                    if res.flux_in and res.flux_in > 1e-12
                    else float("nan")
                )
                rows.append(
                    {
                        "sample_id": sid,
                        "flux_y_scale": fy,
                        "in_monitor_dx": dx,
                        "status": res.status,
                        "R_up": res.split_ratio_upper,
                        "flux_in": res.flux_in,
                        "flux_out_upper": res.flux_out_upper,
                        "flux_out_lower": res.flux_out_lower,
                        "T_total": t_total,
                        "IL_db": res.insertion_loss_db,
                    }
                )
                if res.status == "ok":
                    print(
                        f"{sid} fy={fy} dx={dx}: R_up={res.split_ratio_upper:.4f} "
                        f"T={t_total:.3f} IL={res.insertion_loss_db:.1f}dB"
                    )

    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "flux_il_audit.json"
    out_json.write_text(json.dumps(rows, indent=2))

    lines = [
        "# Flux / IL monitor audit\n",
        "Sweeps `flux_y_scale` (monitor height / wg width) and `in_monitor_dx` (input plane).\n",
        "Goal: check whether low $T_{\\mathrm{total}}$ is monitor placement vs physics.\n\n",
        "| design | fy | dx | R_up | T_total | IL (dB) |\n",
        "|--------|----|----|------|---------|--------|\n",
    ]
    for r in rows:
        if r["status"] != "ok":
            continue
        lines.append(
            f"| `{r['sample_id']}` | {r['flux_y_scale']} | {r['in_monitor_dx']} "
            f"| {r['R_up']:.4f} | {r['T_total']:.3f} | {r['IL_db']:.1f} |\n"
        )
    out_md = OUT / "flux_il_audit.md"
    out_md.write_text("".join(lines))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
