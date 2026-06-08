#!/usr/bin/env python3
"""Mesh convergence sweep for champions (pixel + sdf_geom recipes)."""

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
JSON_PATH = OUT / "champion_mesh_convergence.json"


def run_panel(cfg: dict, resolutions: list[int]) -> list[dict]:
    require_meep()
    rows = []
    samples = list(cfg.get("champions", [])) + list(cfg.get("extended", []))
    for entry in samples:
        sid = entry["id"]
        mask = np.load(REPO / entry["mask"])
        for res in resolutions:
            for recipe_name in ("phase0_v1", "phase0_v1_sdf_geom"):
                extra: dict = {"resolution": res, "mask_flip_y": True}
                if recipe_name == "phase0_v1_sdf_geom":
                    extra["sdf_smooth_um"] = float(cfg.get("promote", {}).get("sdf_smooth_um", 0.04))
                recipe = MeepRecipe.for_version(recipe_name, extra)
                sim = simulate_mask(mask, recipe, verbose=False)
                rows.append(
                    {
                        "sample_id": sid,
                        "recipe": recipe_name,
                        "resolution": res,
                        "status": sim.status,
                        "R_up": float(sim.split_ratio_upper) if sim.status == "ok" else None,
                        "IL_db": float(sim.insertion_loss_db) if sim.status == "ok" else None,
                    }
                )
                if sim.status == "ok":
                    print(f"{sid} {recipe_name} r{res}: R_up={sim.split_ratio_upper:.4f}")
    return rows


def write_summary_md(rows: list[dict], path: Path) -> None:
    champs = sorted({r["sample_id"] for r in rows if r["status"] == "ok"})
    lines = [
        "# Champion mesh convergence\n",
        "Resolutions: 25, 35, 50, 75 px/µm · recipes: `phase0_v1` (pixel) vs `phase0_v1_sdf_geom`\n\n",
        "| Design | pixel span | sdf_geom @ r25–r75 |\n",
        "|--------|------------|--------------------|\n",
    ]
    for sid in champs:
        pix = [r["R_up"] for r in rows if r["sample_id"] == sid and r["recipe"] == "phase0_v1" and r["R_up"] is not None]
        sdf = [r["R_up"] for r in rows if r["sample_id"] == sid and r["recipe"] == "phase0_v1_sdf_geom" and r["R_up"] is not None]
        span = f"{max(pix) - min(pix):.3f}" if pix else "—"
        sdf_s = f"{min(sdf):.3f}–{max(sdf):.3f}" if sdf else "—"
        lines.append(f"| `{sid}` | {span} | {sdf_s} |\n")
    path.write_text("".join(lines))


def plot_rows(rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    all_res: list[int] = []
    all_rup: list[float] = []
    for ax, recipe, title in (
        (axes[0], "phase0_v1", "Pixel grid (phase0_v1)"),
        (axes[1], "phase0_v1_sdf_geom", "SDF audit (phase0_v1_sdf_geom)"),
    ):
        for sid in sorted({r["sample_id"] for r in rows}):
            sub = [
                r
                for r in rows
                if r["sample_id"] == sid and r["recipe"] == recipe and r.get("R_up") is not None
            ]
            if not sub:
                continue
            sub = sorted(sub, key=lambda x: x["resolution"])
            xs = [s["resolution"] for s in sub]
            ys = [s["R_up"] for s in sub]
            all_res.extend(xs)
            all_rup.extend(ys)
            ax.plot(xs, ys, marker="o", ms=4, label=sid)
        ax.axhspan(0.45, 0.55, color="#22c55e", alpha=0.08, zorder=0)
        ax.axhline(0.5, color="gray", ls="--", lw=0.8)
        ax.axhline(0.55, color="gray", ls=":", lw=0.6)
        ax.axhline(0.45, color="gray", ls=":", lw=0.6)
        ax.set_xlabel("Resolution (px/µm)")
        ax.set_ylabel("R_up")
        ax.set_title(title)
        ax.legend(fontsize=5, ncol=2, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        ax.grid(True, alpha=0.3)

    if all_res:
        rmin, rmax = min(all_res), max(all_res)
        pad = max(2, int(0.08 * (rmax - rmin)))
        for ax in axes:
            ax.set_xlim(rmin - pad, rmax + pad)
    ylo = min(all_rup) if all_rup else 0.15
    yhi = max(all_rup) if all_rup else 0.85
    margin = max(0.05, 0.08 * (yhi - ylo))
    for ax in axes:
        ax.set_ylim(max(0.0, min(ylo, 0.45) - margin), min(1.0, max(yhi, 0.55) + margin))

    fig.tight_layout(rect=(0, 0, 0.88, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"champion_mesh_convergence.{ext}", dpi=200)
    plt.close()
    print(f"wrote {OUT / 'champion_mesh_convergence.png'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO / "configs/promote_sdf_geom.yaml")
    p.add_argument("--resolutions", type=str, default="25,35,50,75")
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Plot from existing champion_mesh_convergence.json (no MEEP)",
    )
    p.add_argument("--json", type=Path, default=JSON_PATH, help="JSON for --plot-only")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        if not args.json.exists():
            raise SystemExit(f"Missing {args.json} — run without --plot-only first.")
        rows = json.loads(args.json.read_text())
        plot_rows(rows)
        write_summary_md(rows, OUT / "champion_mesh_convergence.md")
        print(f"wrote {OUT / 'champion_mesh_convergence.md'}")
        return

    cfg = yaml.safe_load(args.config.read_text())
    res = [int(x) for x in args.resolutions.split(",")]
    rows = run_panel(cfg, res)
    JSON_PATH.write_text(json.dumps(rows, indent=2))
    print(f"wrote {JSON_PATH}")

    try:
        plot_rows(rows)
        write_summary_md(rows, OUT / "champion_mesh_convergence.md")
        print(f"wrote {OUT / 'champion_mesh_convergence.md'}")
    except ImportError:
        print("matplotlib not available — skip figures")


if __name__ == "__main__":
    main()
