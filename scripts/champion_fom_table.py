#!/usr/bin/env python3
"""MEEP figures-of-merit for promoted champions (split + transmission + IL)."""

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


def pass_gate(
    r_up: float,
    t_total: float,
    il_db: float,
    *,
    split_tol: float,
    min_t: float,
    max_il: float,
) -> bool:
    if not np.isfinite(r_up) or not np.isfinite(t_total):
        return False
    if abs(r_up - 0.5) > split_tol:
        return False
    if t_total < min_t:
        return False
    if np.isfinite(il_db) and il_db > max_il:
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--split-tol", type=float, default=0.05)
    p.add_argument("--min-transmission", type=float, default=0.80)
    p.add_argument("--max-il-db", type=float, default=2.0)
    args = p.parse_args()

    require_meep()
    cfg = yaml.safe_load(args.config.read_text())
    recipe = MeepRecipe.for_version(
        cfg.get("production_recipe", "phase0_v1"),
        {"resolution": 25, "mask_flip_y": True},
    )

    rows = []
    for entry in cfg.get("champions", []):
        sid = entry["id"]
        mask_path = REPO / entry["mask"]
        mask = np.load(mask_path)
        res = simulate_mask(mask, recipe, verbose=False)
        if res.status != "ok":
            rows.append({"sample_id": sid, "status": res.status, "error": res.error})
            continue
        fi = float(res.flux_in)
        fu = float(res.flux_out_upper)
        fl = float(res.flux_out_lower)
        t_total = (fu + fl) / fi if fi > 0 else float("nan")
        row = {
            "sample_id": sid,
            "mask_path": str(mask_path.relative_to(REPO)),
            "status": "ok",
            "R_up": float(res.split_ratio_upper),
            "split_error": abs(float(res.split_ratio_upper) - 0.5),
            "flux_in": fi,
            "flux_out_upper": fu,
            "flux_out_lower": fl,
            "T_total": t_total,
            "insertion_loss_db": float(res.insertion_loss_db),
            "reflection_note": "not_monitored",
            "pass_release_gate": pass_gate(
                res.split_ratio_upper,
                t_total,
                res.insertion_loss_db,
                split_tol=args.split_tol,
                min_t=args.min_transmission,
                max_il=args.max_il_db,
            ),
        }
        rows.append(row)
        print(
            f"{sid}: R_up={row['R_up']:.4f} T={row['T_total']:.3f} "
            f"IL={row['insertion_loss_db']:.2f} dB pass={row['pass_release_gate']}"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "champion_fom_table.json").write_text(json.dumps(rows, indent=2))

    lines = [
        "# Champion figures of merit @ phase0_v1 r25",
        "",
        f"Gates: |R_up−0.5|≤{args.split_tol}, T_total≥{args.min_transmission}, IL≤{args.max_il_db} dB",
        "",
        "| Design | R_up | |err| | T_total | IL (dB) | Pass? |",
        "|--------|------|-------|---------|---------|-------|",
    ]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"| `{r['sample_id']}` | — | — | — | — | fail ({r.get('error','')}) |")
            continue
        flag = "yes" if r["pass_release_gate"] else "no"
        lines.append(
            f"| `{r['sample_id']}` | {r['R_up']:.4f} | {r['split_error']:.4f} | "
            f"{r['T_total']:.3f} | {r['insertion_loss_db']:.2f} | {flag} |"
        )
    md = OUT / "champion_fom_table.md"
    md.write_text("\n".join(lines) + "\n")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
