#!/usr/bin/env python3
"""Export MEEP-gated search pipeline schematic for preprint Figure 1."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs/preprint/figures"

BOXES = [
    "DRC manifold\nsample",
    "Surrogate\nrank",
    "MEEP\nverify",
    "Corpus /\nmetrics",
]
BOX_COLORS = ["#dbeafe", "#e0e7ff", "#dcfce7", "#fef3c7"]
EDGE = "#334155"


def _add_box(ax, xy, text, color: str) -> FancyBboxPatch:
    w, h = 1.55, 0.72
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=EDGE,
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="semibold",
        color="#0f172a",
        zorder=3,
    )
    return patch


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 1.55))
    ax.set_xlim(0, 8.6)
    ax.set_ylim(0, 1.35)
    ax.axis("off")

    gap = 0.38
    box_w = 1.55
    y = 0.32
    xs = [0.35 + i * (box_w + gap) for i in range(len(BOXES))]

    for i, (label, color) in enumerate(zip(BOXES, BOX_COLORS)):
        _add_box(ax, (xs[i], y), label, color)
        if i < len(BOXES) - 1:
            x0 = xs[i] + box_w + 0.04
            x1 = xs[i + 1] - 0.04
            arrow = FancyArrowPatch(
                (x0, y + 0.36),
                (x1, y + 0.36),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.4,
                color=EDGE,
                zorder=1,
            )
            ax.add_patch(arrow)

    ax.text(
        4.3,
        1.12,
        "MEEP-gated search pipeline",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        4.3,
        0.06,
        "Surrogate narrows candidates; MEEP is the sole promotion authority.",
        ha="center",
        va="center",
        fontsize=8,
        color="#475569",
    )

    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        path = OUT / f"pipeline_schematic.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
