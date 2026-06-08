#!/usr/bin/env python3
"""Aggregate replicate sim-budget runs → mean ± std tables and error-bar plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.pilot import load_pilot_config  # noqa: E402


def load_reports(agg_path: Path, rep_dir: Path) -> list[dict]:
    runs: list[dict] = []
    if agg_path.exists():
        data = json.loads(agg_path.read_text())
        runs.extend(data.get("runs", []))
    if rep_dir.exists():
        for p in sorted(rep_dir.glob("run_*/run_report.json")):
            runs.append(json.loads(p.read_text()))
    # dedupe by replicate_id + seed
    seen: set[tuple] = set()
    out: list[dict] = []
    for r in runs:
        key = (r.get("replicate_id"), r.get("seed"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def reports_to_long_df(runs: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for run in runs:
        rid = run.get("replicate_id")
        seed = run.get("seed")
        for policy, budgets in (run.get("policies") or {}).items():
            for bstr, summary in budgets.items():
                rows.append(
                    {
                        "replicate_id": rid,
                        "seed": seed,
                        "policy": policy,
                        "budget": int(bstr),
                        "n_in_spec": summary.get("n_in_spec"),
                        "n_meep": summary.get("n_meep"),
                        "best_abs_err": summary.get("best_abs_err"),
                        "best_split": summary.get("best_split"),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/wedge_a_production.yaml")
    args = p.parse_args()

    cfg = load_pilot_config(args.config if args.config.is_absolute() else REPO_ROOT / args.config)
    rep_cfg = cfg.get("replication", {})
    agg_path = REPO_ROOT / rep_cfg.get("metrics_aggregate", "data/phase1/wedge_a/sim_budget_replicates.json")
    rep_dir = REPO_ROOT / rep_cfg.get("output_dir", "data/phase1/wedge_a/sim_budget/replicates")
    report_md = REPO_ROOT / rep_cfg.get("report_md", "data/phase1/wedge_a/sim_budget_replication_report.md")

    runs = load_reports(agg_path, rep_dir)
    if not runs:
        raise SystemExit("no replicate reports found")

    long = reports_to_long_df(runs)
    long.to_csv(agg_path.with_suffix(".csv"), index=False)

    budgets = sorted(long["budget"].unique())
    n_rep = int(long["replicate_id"].nunique())

    def _ci95(std: float, n: int) -> float:
        if n < 2 or not np.isfinite(std):
            return float("nan")
        from scipy import stats as sp_stats

        return float(sp_stats.t.ppf(0.975, n - 1) * std / np.sqrt(n))

    stats = (
        long.groupby(["policy", "budget"])
        .agg(
            n_in_spec_mean=("n_in_spec", "mean"),
            n_in_spec_std=("n_in_spec", "std"),
            n_in_spec_ci95=("n_in_spec", lambda s: _ci95(float(s.std(ddof=1)), len(s))),
            best_abs_err_mean=("best_abs_err", "mean"),
            best_abs_err_std=("best_abs_err", "std"),
            best_abs_err_ci95=("best_abs_err", lambda s: _ci95(float(s.std(ddof=1)), len(s))),
            n_replicates=("replicate_id", "nunique"),
        )
        .reset_index()
    )
    stats_path = agg_path.parent / "sim_budget_replication_stats.csv"
    stats.to_csv(stats_path, index=False)

    hdr = " | ".join([f"B={b}" for b in budgets])
    lines = [
        "# Sim-budget replication aggregate\n",
        f"**Replicates:** {n_rep}  \n",
        f"**Policies:** {', '.join(sorted(long['policy'].unique()))}\n\n",
        "## n_in_spec (mean ± std)\n\n",
        f"| Policy | {hdr} |\n",
        f"|--------|{'|'.join(['------'] * len(budgets))}|\n",
    ]
    for policy in sorted(long["policy"].unique()):
        cells = []
        for b in budgets:
            sub = stats[(stats["policy"] == policy) & (stats["budget"] == b)]
            if len(sub):
                m = sub.iloc[0]
                cells.append(f"{m['n_in_spec_mean']:.1f} ± {m['n_in_spec_std']:.1f}")
            else:
                cells.append("—")
        lines.append(f"| `{policy}` | {' | '.join(cells)} |\n")

    lines.append("\n## n_in_spec (95% CI half-width)\n\n")
    lines.append(f"| Policy | {hdr} |\n|--------|{'|'.join(['------'] * len(budgets))}|\n")
    for policy in sorted(long["policy"].unique()):
        cells = []
        for b in budgets:
            sub = stats[(stats["policy"] == policy) & (stats["budget"] == b)]
            if len(sub) and np.isfinite(sub.iloc[0]["n_in_spec_ci95"]):
                cells.append(f"±{sub.iloc[0]['n_in_spec_ci95']:.2f}")
            else:
                cells.append("—")
        lines.append(f"| `{policy}` | {' | '.join(cells)} |\n")

    lines.append("\n## best_abs_err (mean ± std)\n\n")
    lines.append(f"| Policy | {hdr} |\n|--------|{'|'.join(['------'] * len(budgets))}|\n")
    for policy in sorted(long["policy"].unique()):
        cells = []
        for b in budgets:
            sub = stats[(stats["policy"] == policy) & (stats["budget"] == b)]
            if len(sub):
                m = sub.iloc[0]
                cells.append(f"{m['best_abs_err_mean']:.4f} ± {m['best_abs_err_std']:.4f}")
            else:
                cells.append("—")
        lines.append(f"| `{policy}` | {' | '.join(cells)} |\n")

    paired_path = agg_path.parent / "sim_budget_paired_wins.md"
    pairs = [
        ("surrogate_rank", "sigma_meep", "n_in_spec", "higher"),
        ("surrogate_rank", "hierarchical_35", "n_in_spec", "higher"),
        ("surrogate_rank", "sigma_meep", "best_abs_err", "lower"),
    ]
    plines = ["# Paired seed comparisons\n", f"**Replicates:** {n_rep}\n\n"]
    for a, b, metric, better in pairs:
        if a not in long["policy"].values or b not in long["policy"].values:
            continue
        plines.append(f"## `{a}` vs `{b}` on `{metric}` ({better} is better)\n\n")
        plines.append("| Budget | A wins | B wins | ties | A mean | B mean |\n")
        plines.append("|--------|--------|--------|------|--------|--------|\n")
        for bud in budgets:
            sub = long[long["budget"] == bud]
            piv = sub.pivot_table(
                index="replicate_id", columns="policy", values=metric, aggfunc="first"
            )
            if a not in piv.columns or b not in piv.columns:
                continue
            va, vb = piv[a], piv[b]
            mask = va.notna() & vb.notna()
            va, vb = va[mask], vb[mask]
            if better == "higher":
                aw = int((va > vb).sum())
                bw = int((vb > va).sum())
            else:
                aw = int((va < vb).sum())
                bw = int((vb < va).sum())
            ties = int(len(va) - aw - bw)
            plines.append(
                f"| {bud} | {aw}/{len(va)} | {bw}/{len(va)} | {ties} | "
                f"{va.mean():.3f} | {vb.mean():.3f} |\n"
            )
        plines.append("\n")
    paired_path.write_text("".join(plines))
    lines.append(f"\nPaired comparisons: `{paired_path.relative_to(REPO_ROOT)}`\n")

    report_md.write_text("".join(lines))

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        policies = sorted(long["policy"].unique())
        budgets = sorted(long["budget"].unique())
        x = np.arange(len(budgets))
        width = 0.8 / max(len(policies), 1)

        for i, policy in enumerate(policies):
            means, stds = [], []
            for b in budgets:
                sub = stats[(stats["policy"] == policy) & (stats["budget"] == b)]
                means.append(float(sub["n_in_spec_mean"].iloc[0]) if len(sub) else 0)
                stds.append(float(sub["n_in_spec_std"].iloc[0]) if len(sub) else 0)
            off = (i - len(policies) / 2) * width + width / 2
            axes[0].bar(x + off, means, width, yerr=stds, label=policy, capsize=3)

        axes[0].set_xticks(x)
        axes[0].set_xticklabels([str(b) for b in budgets])
        axes[0].set_xlabel("MEEP budget B")
        axes[0].set_ylabel("n_in_spec")
        axes[0].set_title("In-spec count (mean ± std over replicates)")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)

        # best_abs_err line plot
        for policy in policies:
            means, stds = [], []
            for b in budgets:
                sub = stats[(stats["policy"] == policy) & (stats["budget"] == b)]
                means.append(float(sub["best_abs_err_mean"].iloc[0]) if len(sub) else np.nan)
                stds.append(float(sub["best_abs_err_std"].iloc[0]) if len(sub) else 0)
            axes[1].errorbar(budgets, means, yerr=stds, marker="o", label=policy, capsize=3)
        axes[1].set_xlabel("MEEP budget B")
        axes[1].set_ylabel("|split − 0.5| (best in run)")
        axes[1].set_title("Best absolute error")
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        png = agg_path.parent / "sim_budget_replication_errorbars.png"
        fig.savefig(png, dpi=160)
        plt.close()
        print(f"wrote {png}")
    except ImportError:
        pass

    print(stats.to_string(index=False))
    print(f"wrote {stats_path}")
    print(f"wrote {report_md}")


if __name__ == "__main__":
    main()
