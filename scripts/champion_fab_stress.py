#!/usr/bin/env python3
"""Fabrication stress: binary morphology on upsampled mask grid (±10/20/30 nm)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nano_inv.fab_stress import (  # noqa: E402
    GRID,
    ISLAND_UM,
    nm_per_px,
    stress_radius_px,
    stress_variants,
)
from nano_inv.meep_sim import MeepRecipe, require_meep, simulate_mask  # noqa: E402

OUT = REPO / "data/phase1/release"
DEFAULT_CFG = REPO / "configs/promote_sdf_geom.yaml"
STRESS_NM = (10, 20, 30)
DELTA_FLAG = 0.05
DEFAULT_UPSCALE = 5


def plot_fab_stress(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    champs = sorted({r["sample_id"] for r in rows if r.get("status") == "ok"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, ops, title in (
        (axes[0], ("erode",), "Erosion stress"),
        (axes[1], ("dilate",), "Dilation stress"),
    ):
        for sid in champs:
            pts = []
            for r in rows:
                if r.get("sample_id") != sid or r.get("status") != "ok":
                    continue
                if r.get("operation") not in ops and r.get("variant") != "nominal":
                    continue
                if r.get("variant") == "nominal":
                    pts.append((0, r["R_up"]))
                else:
                    pts.append((r.get("stress_nm", 0), r["R_up"]))
            if not pts:
                continue
            pts = sorted(pts)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=4, label=sid)
        ax.axhspan(0.45, 0.55, color="#22c55e", alpha=0.08, zorder=0)
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.axhline(0.55, color="gray", ls=":", lw=0.6)
        ax.axhline(0.45, color="gray", ls=":", lw=0.6)
        ax.set_xlabel("Morph stress (nm)")
        ax.set_ylabel("R_up")
        ax.set_title(title)
        ax.set_xlim(-2, 35)
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=5, ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0, 0.88, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"champion_fab_stress.{ext}", dpi=200)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--delta-flag", type=float, default=DELTA_FLAG)
    p.add_argument(
        "--morph-upscale",
        type=int,
        default=DEFAULT_UPSCALE,
        help="Upsample masks by this factor before morphology (default 5 → ~4.4 nm/px fine grid)",
    )
    p.add_argument("--stress-nm", type=str, default="10,20,30", help="Comma-separated stress levels in nm")
    p.add_argument("--dry-run", action="store_true", help="List stress variants; skip MEEP")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Plot from existing champion_fab_stress.json (no MEEP)",
    )
    p.add_argument("--json", type=Path, default=OUT / "champion_fab_stress.json")
    args = p.parse_args()

    if args.plot_only:
        if not args.json.exists():
            raise SystemExit(f"Missing {args.json}")
        rows = json.loads(args.json.read_text())
        plot_fab_stress(rows)
        print(f"wrote {OUT / 'champion_fab_stress.png'}")
        return

    stress_levels = tuple(int(x.strip()) for x in args.stress_nm.split(",") if x.strip())
    upscale = max(1, args.morph_upscale)
    coarse_nm = nm_per_px(upscale=1)
    fine_nm = nm_per_px(upscale=upscale)

    cfg = yaml.safe_load(args.config.read_text())
    recipe = None
    if not args.dry_run:
        require_meep()
        recipe = MeepRecipe.for_version(
            cfg.get("production_recipe", "phase0_v1"),
            {"resolution": 25, "mask_flip_y": True},
        )

    rows: list[dict] = []
    for entry in cfg.get("champions", []):
        sid = entry["id"]
        mask = np.load(REPO / entry["mask"])
        nominal_r_up: float | None = None
        nominal_err: float | None = None

        for vid, stressed, r_px, op, eff_nm in stress_variants(
            mask, stress_levels, upscale=upscale
        ):
            row: dict = {
                "sample_id": sid,
                "variant": vid,
                "operation": op,
                "stress_nm": 0 if vid == "nominal" else int(vid.split("_")[1].replace("nm", "")),
                "radius_px_fine": r_px,
                "effective_nm": eff_nm,
                "morph_upscale": upscale,
                "nm_per_px_coarse": coarse_nm,
                "nm_per_px_fine": fine_nm,
            }
            if args.dry_run:
                row["status"] = "dry_run"
                rows.append(row)
                print(
                    f"{sid} {vid}: r_fine={r_px}px "
                    f"(≈{eff_nm:.1f} nm @ {fine_nm:.2f} nm/px, upscale={upscale})"
                )
                continue

            res = simulate_mask(stressed, recipe, verbose=False)
            if res.status != "ok":
                row.update({"status": res.status, "error": res.error})
                rows.append(row)
                continue

            r_up = float(res.split_ratio_upper)
            split_err = abs(r_up - 0.5)
            row.update({"status": "ok", "R_up": r_up, "split_error": split_err})
            if vid == "nominal":
                nominal_r_up = r_up
                nominal_err = split_err
            elif nominal_r_up is not None:
                delta = r_up - nominal_r_up
                row["delta_R_up"] = delta
                row["delta_split_error"] = split_err - (nominal_err or split_err)
                row["flag_delta_R_up"] = abs(delta) > args.delta_flag
            rows.append(row)
            flag = row.get("flag_delta_R_up")
            extra = f" ΔR={row.get('delta_R_up', 0):+.4f} FLAG={flag}" if vid != "nominal" else ""
            print(f"{sid} {vid}: R_up={r_up:.4f} err={split_err:.4f}{extra}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "champion_fab_stress.json").write_text(json.dumps(rows, indent=2))

    lines = [
        "# Champion fabrication stress (binary morphology)",
        "",
        f"Mask grid: {GRID}×{GRID} on {ISLAND_UM} µm island → {coarse_nm:.2f} nm/px (coarse)",
        f"Morph upscale: **{upscale}×** → {GRID * upscale}×{GRID * upscale} fine grid @ {fine_nm:.2f} nm/px",
        f"Stress levels: ±{', ±'.join(str(n) for n in stress_levels)} nm (disk SE on fine grid)",
        f"MEEP: phase0_v1 r25, mask_flip_y=True · flag if |ΔR_up| > {args.delta_flag}",
        "",
        "### Nominal radius mapping (fine grid)",
        "",
        "| stress (nm) | radius (px) | effective (nm) |",
        "|-------------|-------------|----------------|",
    ]
    for nm in stress_levels:
        r = stress_radius_px(float(nm), upscale=upscale)
        lines.append(f"| {nm} | {r} | {r * fine_nm:.1f} |")
    lines.append("")

    if args.dry_run:
        lines.append("**Dry run** — variants listed; no MEEP simulations.\n")
    lines.extend(
        ["| Design | Variant | R_up | |err| | ΔR_up | Flag |", "|--------|---------|------|-------|-------|------|"]
    )
    for r in rows:
        if r.get("status") == "dry_run":
            lines.append(f"| `{r['sample_id']}` | {r['variant']} | — | — | — | dry-run |")
            continue
        if r.get("status") != "ok":
            lines.append(f"| `{r['sample_id']}` | {r['variant']} | — | — | — | fail |")
            continue
        d = r.get("delta_R_up")
        d_s = f"{d:+.4f}" if d is not None else "—"
        flag = "yes" if r.get("flag_delta_R_up") else ("no" if d is not None else "—")
        lines.append(
            f"| `{r['sample_id']}` | {r['variant']} | {r['R_up']:.4f} | {r['split_error']:.4f} | {d_s} | {flag} |"
        )
    md = OUT / "champion_fab_stress.md"
    md.write_text("\n".join(lines) + "\n")
    print(f"wrote {md}")
    try:
        plot_fab_stress(rows)
        print(f"wrote {OUT / 'champion_fab_stress.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
