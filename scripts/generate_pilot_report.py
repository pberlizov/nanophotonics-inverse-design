#!/usr/bin/env python3
"""Generate pilot sim-budget report (Markdown + JSON) from wedge_a metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.pilot import (  # noqa: E402
    as_repo_relative,
    build_template_context,
    deliverables_dir,
    latest_sim_budget_run,
    load_metrics,
    load_pilot_config,
    policy_summary_table,
    render_template,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/pilot/benchmark_50_50.yaml")
    return p.parse_args()


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "_No sim-budget run found. Run `bash scripts/run_pilot.sh --full-meep`._\n"
    lines = [
        "| Policy | B | Best split | |err| | In-spec | Best ID |",
        "|--------|---|------------|-------|---------|---------|",
    ]
    for r in sorted(rows, key=lambda x: (x["budget"], x["policy"])):
        lines.append(
            f"| `{r['policy']}` | {r['budget']} | {r.get('best_split', float('nan')):.3f} "
            f"| {r.get('best_abs_err', float('nan')):.3f} | {r.get('n_in_spec', 0)}/{r.get('n_meep', 0)} "
            f"| `{r.get('best_sample_id', '')}` |"
        )
    return "\n".join(lines) + "\n"


def ranking_section(cfg: dict) -> str:
    src = cfg.get("sources", {})
    path = REPO_ROOT / src.get("ranking_eval", "data/phase1/wedge_a/ranking_eval.json")
    if not path.exists():
        return "_Ranking eval not found._\n"
    ev = json.loads(path.read_text())
    return (
        f"- **Ranking wins (holdout):** `{ev.get('ranking_wins')}`\n"
        f"- **Top-k:** {ev.get('top_k')} on {ev.get('n_samples')} perturb labels\n"
        f"- **Mean |err| surrogate top-k:** {ev.get('mean_abs_err_surrogate_topk', '?'):.3f}\n"
        f"- **Mean |err| random top-k:** {ev.get('mean_abs_err_random_topk', '?'):.3f}\n"
        f"- **In-spec in surrogate top-k:** {ev.get('n_in_spec_surrogate_topk')} vs random {ev.get('n_in_spec_random_topk')}\n"
    )


def claims_section(rows: list[dict], target: float, tol: float) -> str:
    if not rows:
        return (
            "## Qualified claims (pending data)\n\n"
            "Run the full sim-budget study (budgets 30, 50, 100) before external claims.\n"
        )
    by_budget: dict[int, list[dict]] = {}
    for r in rows:
        by_budget.setdefault(r["budget"], []).append(r)

    lines = ["## Qualified claims (use in outreach)\n"]
    for budget in sorted(by_budget):
        group = by_budget[budget]
        sur = next((x for x in group if x["policy"] == "surrogate_rank"), None)
        rnd = next((x for x in group if x["policy"] == "random_perturb"), None)
        if sur and rnd:
            lines.append(
                f"- At **MEEP budget B={budget}**, surrogate-ranked verification found "
                f"**{sur.get('n_in_spec', 0)}** in-spec design(s) vs **{rnd.get('n_in_spec', 0)}** for random perturbation "
                f"(target {target:.2f} ± {tol:.2f}).\n"
            )
    lines.append(
        "\n**Primary pitch:** functional + structural novelty vs published reference "
        "(see `data/phase1/novelty/novelty_summary.md`), not surrogate R².\n"
        "\n**Not claimed:** foundry yield, PDK sign-off, or that σ-local champions are unrelated shapes.\n"
    )
    return "".join(lines)


def main() -> None:
    args = parse_args()
    cfg = load_pilot_config(args.config)
    out = deliverables_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(cfg)
    run = latest_sim_budget_run(metrics)
    rows = policy_summary_table(run) if run else []
    target = float((run or {}).get("target", cfg["targets"]["split_ratio_1550"]))
    tol = float((run or {}).get("tolerance", cfg.get("sim_budget", {}).get("tolerance", 0.05)))

    ctx = build_template_context(cfg)
    md_parts = [
        f"# Sim-budget report — {ctx['PILOT_TITLE']}\n",
        f"**Pilot ID:** `{ctx['PILOT_ID']}`  \n",
        f"**Date:** {ctx['CONTRACT_DATE']}\n",
        "## Target\n",
        f"- Split ratio @ {ctx['WAVELENGTH_NM']} nm: **{target:.2f}** ± **{tol:.2f}**\n",
        f"- In-spec: {ctx['IN_SPEC_DEFINITION']}\n",
        f"- MEEP recipe: `{ctx['RECIPE_VERSION']}` (resolution {ctx['MEEP_RESOLUTION']})\n",
        "## Sim-budget comparison\n",
        format_table(rows),
        "## Surrogate ranking (holdout)\n",
        ranking_section(cfg),
        claims_section(rows, target, tol),
        "## Champions (existence proof)\n",
    ]
    for ch in cfg.get("champions") or []:
        split = ch.get("split_ratio_upper")
        md_parts.append(f"- `{ch['sample_id']}`: MEEP split **{split}** — {ch.get('source', '')}\n")

    md_parts.append(
        "\n## Next steps for external pilot\n\n"
        "1. Run `--full-meep` sim-budget (30/50/100) if not yet done.\n"
        "2. Customize `configs/pilot/<client>.yaml` with client spec.\n"
        "3. Attach `deliverables/design_dossier.csv` + `SIM_CONTRACT.md` to proposal.\n"
    )

    report_md = "".join(md_parts)
    (out / "SIM_BUDGET_REPORT.md").write_text(report_md)

    report_json = {
        "pilot_id": ctx["PILOT_ID"],
        "target": target,
        "tolerance": tol,
        "policies": rows,
        "budgets_in_run": sorted({r["budget"] for r in rows}),
        "full_budgets_recommended": cfg.get("sim_budget", {}).get("budgets", [30, 50, 100]),
    }
    (out / "sim_budget_report.json").write_text(json.dumps(report_json, indent=2))
    print(f"wrote {out / 'SIM_BUDGET_REPORT.md'}")
    print(f"wrote {out / 'sim_budget_report.json'}")


if __name__ == "__main__":
    main()
