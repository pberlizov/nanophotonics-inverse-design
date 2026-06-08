#!/usr/bin/env python3
"""C-band wavelength sweep for promoted champions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nano_inv.meep_sim import MeepRecipe, require_meep, simulate_mask_broadband  # noqa: E402

OUT = REPO / "data/phase1/release"
DEFAULT_CFG = REPO / "configs/promote_sdf_geom.yaml"
TARGET_SPLIT = 0.5


def write_summary_md(
    path: Path,
    *,
    results: list[dict],
    wls: list[float],
    max_worst_err: float,
) -> None:
    lines = [
        "# Champion C-band broadband sweep\n",
        f"λ grid: {wls[0]:.3f}–{wls[-1]:.3f} µm ({len(wls)} points"
        + (
            f", step {wls[1] - wls[0]:.3f} µm)\n"
            if len(wls) > 1
            else ")\n"
        ),
        f"**Release gate:** worst |R_up − {TARGET_SPLIT}| ≤ {max_worst_err:.2f} over band\n",
        "\n| Design | R_up @ 1.55 | worst |err| | mean IL (dB) | pass gate? |\n",
        "|--------|-------------|-------------|--------------|----------|\n",
    ]
    n_pass = 0
    for r in results:
        if r["status"] != "ok":
            lines.append(f"| {r['sample_id']} | — | — | — | error |\n")
            continue
        by_wl = r["R_up_by_wl"]
        r155 = by_wl.get("1.55")
        if r155 is None:
            for k, v in by_wl.items():
                if abs(float(k) - 1.55) < 1e-6:
                    r155 = v
                    break
        if r155 is None:
            r155 = float("nan")
        passed = r.get("pass_broadband_gate", False)
        n_pass += int(passed)
        lines.append(
            f"| {r['sample_id']} | {float(r155):.4f} | {r['worst_split_error']:.4f} "
            f"| {r['mean_IL_db']:.2f} | {'yes' if passed else 'no'} |\n"
        )
    lines.append(f"\n**Summary:** {n_pass}/{len(results)} pass broadband gate.\n")
    lines.append(
        "\n> Current champions are narrowband-tuned at 1550 nm; expect failure until "
        "broadband-aware search (`meep_search_local.py --objective broadband`) finds new candidates.\n"
    )
    path.write_text("".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--wl-start", type=float, default=1.53)
    p.add_argument("--wl-stop", type=float, default=1.57)
    p.add_argument("--wl-step", type=float, default=0.005)
    p.add_argument(
        "--max-worst-split-error",
        type=float,
        default=0.05,
        help="Release gate: max |R_up - 0.5| over the sweep",
    )
    args = p.parse_args()

    require_meep()
    cfg = yaml.safe_load(args.config.read_text())
    base = MeepRecipe.for_version(
        cfg.get("production_recipe", "phase0_v1"),
        {"resolution": 25, "mask_flip_y": True},
    )
    wls = list(np.arange(args.wl_start, args.wl_stop + 1e-9, args.wl_step))
    wls = [round(float(w), 3) for w in wls]

    results = []
    for entry in cfg.get("champions", []):
        sid = entry["id"]
        mask = np.load(REPO / entry["mask"])
        bb = simulate_mask_broadband(mask, base, wls, target_split=0.5, verbose=False)
        passed = (
            bb.status == "ok"
            and np.isfinite(bb.worst_split_error)
            and bb.worst_split_error <= args.max_worst_split_error
        )
        results.append(
            {
                "sample_id": sid,
                "wavelengths_um": wls,
                "status": bb.status,
                "R_up_by_wl": {str(k): v for k, v in bb.splits_by_wavelength.items()},
                "IL_by_wl": {str(k): v for k, v in bb.insertion_loss_by_wavelength.items()},
                "worst_split_error": bb.worst_split_error,
                "mean_IL_db": bb.mean_insertion_loss_db,
                "pass_broadband_gate": passed,
                "max_worst_split_error": args.max_worst_split_error,
                "error": bb.error,
            }
        )
        gate = "PASS" if passed else "FAIL"
        print(
            f"{sid}: worst |err|={bb.worst_split_error:.4f} mean_IL={bb.mean_insertion_loss_db:.2f} [{gate}]"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "champion_broadband.json").write_text(json.dumps(results, indent=2))
    write_summary_md(
        OUT / "champion_broadband.md",
        results=results,
        wls=wls,
        max_worst_err=args.max_worst_split_error,
    )
    print(f"wrote {OUT / 'champion_broadband.md'}")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for r in results:
            if r["status"] != "ok":
                continue
            xs = sorted(float(x) for x in r["R_up_by_wl"].keys())
            axes[0].plot(xs, [r["R_up_by_wl"][str(x)] for x in xs], marker="o", label=r["sample_id"])
            axes[1].plot(xs, [r["IL_by_wl"][str(x)] for x in xs], marker="o", label=r["sample_id"])
        axes[0].axhline(0.5, color="gray", ls="--", lw=0.8)
        axes[0].axhline(0.55, color="gray", ls=":", lw=0.6)
        axes[0].axhline(0.45, color="gray", ls=":", lw=0.6)
        axes[0].set_xlabel("λ (µm)")
        axes[0].set_ylabel("R_up")
        axes[0].set_title("Split ratio vs wavelength")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel("λ (µm)")
        axes[1].set_ylabel("IL (dB)")
        axes[1].set_title("Insertion loss vs wavelength")
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"champion_broadband.{ext}", dpi=200)
        plt.close()
        print(f"wrote {OUT / 'champion_broadband.png'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
