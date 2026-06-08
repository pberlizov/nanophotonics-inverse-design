#!/usr/bin/env python3
"""Export figures for docs/preprint/manuscript.tex."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NOVELTY_CSV = REPO / "data/phase1/novelty/novelty_metrics.csv"
STATS_CSV = REPO / "data/phase1/wedge_a/sim_budget_replication_stats.csv"
OUT = REPO / "docs/preprint/figures"

# Match manuscript table (five primary replicates; exclude run_06 from plot if present).
PRIMARY_REPLICATES = 6

POLICY_LABELS = {
    "surrogate_rank": "surrogate_rank",
    "hierarchical_35": "hierarchical_35",
    "hierarchical_50": "hierarchical_50",
    "hierarchical_65": "hierarchical_65",
    "random_perturb": "random_perturb",
    "sigma_meep": "sigma_meep",
}

POLICY_ORDER = list(POLICY_LABELS.keys())


def _cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(values)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def plot_hamming_cdf() -> None:
    df = pd.read_csv(NOVELTY_CSV)
    perturb = df[df["category"] == "corpus_perturbation"]
    perlin = df[df["category"] == "corpus_perlin"]
    perturb_ins = perturb[perturb["in_spec"] == True]
    perlin_ins = perlin[perlin["in_spec"] == True]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    if len(perturb):
        x, y = _cdf(perturb["hamming_fraction"].to_numpy())
        ax.plot(x, y, color="#1f77b4", lw=2, label=f"σ-perturb (n={len(perturb)})")
    if len(perturb_ins):
        x, y = _cdf(perturb_ins["hamming_fraction"].to_numpy())
        ax.plot(
            x,
            y,
            color="#1f77b4",
            lw=1.5,
            ls="--",
            label=f"σ-perturb in-spec (n={len(perturb_ins)})",
        )
    if len(perlin_ins):
        x, y = _cdf(perlin_ins["hamming_fraction"].to_numpy())
        ax.plot(
            x,
            y,
            color="#ff7f0e",
            lw=2,
            label=f"Perlin in-spec (n={len(perlin_ins)})",
        )

    champions = [
        ("local_00022", "#d62728", "local_00022"),
        ("meep_bo_00128", "#8c564b", "meep_bo_00128"),
    ]
    for sid, color, label in champions:
        row = df[df["sample_id"] == sid]
        if len(row):
            ax.axvline(
                float(row["hamming_fraction"].iloc[0]),
                color=color,
                ls=":",
                lw=1.5,
                label=label,
            )

    ax.axvline(0.06, color="gray", ls="-.", lw=1, alpha=0.7, label="expert σ-ball (~6%)")
    ax.set_xlim(0, max(0.65, float(perturb["hamming_fraction"].max()) * 1.05))
    ax.set_xlabel("Pixel Hamming fraction vs. ref_published")
    ax.set_ylabel("CDF")
    ax.set_title("Mask distance from published reference")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = OUT / f"hamming_cdf.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def plot_sim_budget() -> None:
    stats = pd.read_csv(STATS_CSV)
    stats = stats[stats["n_replicates"] == PRIMARY_REPLICATES].copy()
    budgets = sorted(stats["budget"].unique())
    policies = [p for p in POLICY_ORDER if p in set(stats["policy"])]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    x = np.arange(len(budgets))
    width = 0.8 / max(len(policies), 1)

    colors = plt.cm.tab10(np.linspace(0, 0.9, len(policies)))
    bar_tops: list[float] = []
    err_tops: list[float] = []
    for i, policy in enumerate(policies):
        sub = stats[stats["policy"] == policy]
        means, stds = [], []
        for b in budgets:
            row = sub[sub["budget"] == b]
            means.append(float(row["n_in_spec_mean"].iloc[0]) if len(row) else 0)
            stds.append(float(row["n_in_spec_std"].iloc[0]) if len(row) else 0)
        bar_tops.extend(m + s for m, s in zip(means, stds))
        off = (i - len(policies) / 2) * width + width / 2
        axes[0].bar(
            x + off,
            means,
            width,
            yerr=stds,
            label=POLICY_LABELS[policy],
            capsize=2,
            color=colors[i],
            edgecolor="white",
            linewidth=0.5,
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(int(b)) for b in budgets])
    axes[0].set_xlim(-0.5, len(budgets) - 0.5)
    axes[0].set_xlabel("MEEP budget $B$")
    axes[0].set_ylabel("In-spec count (mean ± std)")
    axes[0].set_title(f"In-spec yield ($n={PRIMARY_REPLICATES}$ replicates)")
    if bar_tops:
        axes[0].set_ylim(0, max(bar_tops) * 1.15)
    axes[0].legend(fontsize=6.5, ncol=2, loc="upper left")
    axes[0].grid(True, axis="y", alpha=0.25)

    for i, policy in enumerate(policies):
        sub = stats[stats["policy"] == policy]
        means, stds = [], []
        for b in budgets:
            row = sub[sub["budget"] == b]
            m = float(row["best_abs_err_mean"].iloc[0]) if len(row) else np.nan
            s = float(row["best_abs_err_std"].iloc[0]) if len(row) else 0
            means.append(m)
            stds.append(s)
            if np.isfinite(m):
                err_tops.append(m + s)
        axes[1].errorbar(
            budgets,
            means,
            yerr=stds,
            marker="o",
            lw=1.5,
            capsize=3,
            label=POLICY_LABELS[policy],
            color=colors[i],
        )

    axes[1].set_xticks(budgets)
    axes[1].set_xlim(min(budgets) - 5, max(budgets) + 5)
    axes[1].set_xlabel("MEEP budget $B$")
    axes[1].set_ylabel(r"Best $|R_{\mathrm{up}} - 0.5|$")
    axes[1].set_title("Best single design per run")
    if err_tops:
        axes[1].set_ylim(0, max(err_tops) * 1.2)
    axes[1].legend(fontsize=6.5, ncol=2, loc="upper right")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    for name in ("sim_budget_curve", "sim_budget_replication_errorbars"):
        for ext in ("pdf", "png"):
            path = OUT / f"{name}.{ext}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not NOVELTY_CSV.exists():
        raise SystemExit(f"missing {NOVELTY_CSV} — run characterize_design_novelty.py")
    if not STATS_CSV.exists():
        raise SystemExit(f"missing {STATS_CSV} — run aggregate_sim_budget_replicates.py")
    plot_hamming_cdf()
    plot_sim_budget()


if __name__ == "__main__":
    main()
