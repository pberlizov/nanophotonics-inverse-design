#!/usr/bin/env python3
"""Scatter split_err vs IL_db for loss-aware hunt (+ champion baselines)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "data/phase1/loss_aware_hunt/loss_aware_trials.csv"
P2_CSV = REPO / "data/phase1/phase2_il_hunt/phase2_il_trials.csv"
OUT = REPO / "docs/preprint/figures"

BASELINES = {
    "cand_000261": (0.005, 19.4),
    "local_00022": (0.001, 17.8),
    "meep_bo_00093": (0.003, 16.7),
}


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"Missing {CSV}")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ins = df[df["in_spec"] == True]
    out = df[df["in_spec"] != True]
    if len(out):
        ax.scatter(out["split_err"], out["IL_db"], c="#aec7e8", s=28, alpha=0.7, label="out of split spec")
    if len(ins):
        ax.scatter(ins["split_err"], ins["IL_db"], c="#1f77b4", s=40, alpha=0.9, label="in-spec split")

    for sid, (se, il) in BASELINES.items():
        ax.scatter([se], [il], marker="*", s=120, zorder=5, label=f"{sid} (split-only)")

    if P2_CSV.exists():
        p2 = pd.read_csv(P2_CSV)
        p2_ins = p2[p2["in_spec"] == True]
        p2_out = p2[p2["in_spec"] != True]
        if len(p2_out):
            ax.scatter(
                p2_out["split_err"],
                p2_out["IL_db"],
                c="#ff7f0e",
                s=22,
                alpha=0.45,
                marker="x",
                label="Phase 2 IL (out of split spec)",
            )
        if len(p2_ins):
            ax.scatter(
                p2_ins["split_err"],
                p2_ins["IL_db"],
                c="#d62728",
                s=36,
                alpha=0.85,
                marker="D",
                label="Phase 2 IL (in-spec)",
            )
        best = p2.sort_values("IL_db").iloc[0]
        ax.annotate(
            f"{best['IL_db']:.1f} dB",
            (best["split_err"], best["IL_db"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color="#d62728",
        )

    ax.axvline(0.05, color="gray", ls="--", lw=1, alpha=0.6)
    ax.axhline(12.0, color="orange", ls=":", lw=1, alpha=0.7, label="IL soft cap (12 dB)")

    all_x = list(df["split_err"])
    all_y = list(df["IL_db"])
    if P2_CSV.exists():
        all_x.extend(p2["split_err"].tolist())
        all_y.extend(p2["IL_db"].tolist())
    x_max = max(max(all_x, default=0.08), 0.08)
    y_min = min(all_y, default=5.0)
    y_max = max(all_y, default=22.0)
    ax.set_xlim(0, x_max * 1.08)
    ax.set_ylim(max(5.0, y_min - 1.0), min(24.0, y_max + 1.5))

    ax.set_xlabel(r"Split error $|R_{\mathrm{up}} - 0.5|$")
    ax.set_ylabel("Insertion loss (dB)")
    ax.set_title("Split vs. IL (Phase B + Phase 2)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = OUT / f"loss_aware_tradeoff.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
