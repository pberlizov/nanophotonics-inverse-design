#!/usr/bin/env python3
"""Render sim-budget curve PNG for outreach slide deck."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.pilot import (  # noqa: E402
    deliverables_dir,
    latest_sim_budget_run,
    load_metrics,
    load_pilot_config,
    policy_summary_table,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/pilot/benchmark_50_50.yaml")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_pilot_config(args.config)
    out = deliverables_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(cfg)
    run = latest_sim_budget_run(metrics)
    rows = policy_summary_table(run) if run else []
    if not rows:
        print("no sim-budget data; skip plot")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip plot")
        return

    target = float(run.get("target", 0.5))
    policies = sorted({r["policy"] for r in rows})
    budgets = sorted({r["budget"] for r in rows})

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for policy in policies:
        xs, ys, ins = [], [], []
        for b in budgets:
            match = [r for r in rows if r["policy"] == policy and r["budget"] == b]
            if not match:
                continue
            m = match[0]
            xs.append(b)
            ys.append(m.get("best_abs_err", float("nan")))
            ins.append(m.get("n_in_spec", 0))
        if xs:
            axes[0].plot(xs, ys, marker="o", label=policy.replace("_", " "))
            axes[1].plot(xs, ins, marker="s", label=policy.replace("_", " "))

    axes[0].axhline(float(cfg.get("sim_budget", {}).get("tolerance", 0.05)), color="gray", ls="--", label="tolerance")
    axes[0].set_xlabel("MEEP budget B")
    axes[0].set_ylabel(f"|split − {target:.2f}| (best in run)")
    axes[0].set_title("Best absolute error vs budget")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("MEEP budget B")
    axes[1].set_ylabel("In-spec count")
    axes[1].set_title("In-spec designs vs budget")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Sim-budget study — {cfg.get('pilot', {}).get('id', 'pilot')}", fontsize=11)
    fig.tight_layout()
    png = out / "sim_budget_curve.png"
    fig.savefig(png, dpi=160)
    plt.close()
    print(f"wrote {png}")

    slide = out / "SLIDE_SIM_BUDGET.md"
    slide.write_text(
        "# One-slide summary\n\n"
        f"![Sim-budget curves](sim_budget_curve.png)\n\n"
        "**Talk track:** At equal MEEP budget, we compare random on-manifold search, "
        "σ-only MEEP optimization, and surrogate-ranked verification. "
        "Promotion always requires MEEP — the surrogate only shortlists.\n"
    )


if __name__ == "__main__":
    main()
