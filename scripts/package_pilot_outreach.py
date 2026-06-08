#!/usr/bin/env python3
"""Bundle deliverables + docs into outreach folder with README index."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.pilot import (  # noqa: E402
    as_repo_relative,
    build_template_context,
    deliverables_dir,
    load_pilot_config,
    render_template,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/pilot/benchmark_50_50.yaml")
    return p.parse_args()


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    cfg = load_pilot_config(args.config)
    dlv = deliverables_dir(cfg)
    outreach = resolve_path(cfg.get("outreach", {}).get("dir", f"data/pilot/{cfg['pilot']['id']}/outreach"))
    outreach.mkdir(parents=True, exist_ok=True)

    for name in (
        "SIM_BUDGET_REPORT.md",
        "SIM_CONTRACT.md",
        "design_dossier.csv",
        "sim_budget_curve.png",
        "SLIDE_SIM_BUDGET.md",
        "manifest.json",
    ):
        copy_if_exists(dlv / name, outreach / name)

    novelty = REPO_ROOT / "data/phase1/novelty"
    for name in ("novelty_summary.md", "novelty_report.json", "hamming_cdf.png", "novelty_metrics.csv"):
        copy_if_exists(novelty / name, outreach / name)
    copy_if_exists(REPO_ROOT / "docs/VALUE_PROPOSITION.md", outreach / "VALUE_PROPOSITION.md")
    rep_md = (cfg.get("sources") or {}).get("sim_budget_replication")
    if rep_md:
        copy_if_exists(REPO_ROOT / rep_md, outreach / Path(rep_md).name)
    copy_if_exists(
        REPO_ROOT / "docs/SIM_BUDGET_REPLICATION_RESULTS.md",
        outreach / "SIM_BUDGET_REPLICATION_RESULTS.md",
    )
    crosscheck = REPO_ROOT / "data/phase1/crosscheck"
    for name in (
        "crosscheck_report.md",
        "crosscheck_report.json",
        "crosscheck_results.csv",
        "LUMERICAL_MANUAL.md",
    ):
        copy_if_exists(crosscheck / name, outreach / name)
    copy_if_exists(
        REPO_ROOT / "data/phase1/wedge_a/sim_budget_replication_errorbars.png",
        outreach / "sim_budget_replication_errorbars.png",
    )

    designs_src = dlv / "designs"
    designs_dst = outreach / "designs"
    if designs_src.exists():
        if designs_dst.exists():
            shutil.rmtree(designs_dst)
        shutil.copytree(designs_src, designs_dst)

    docs_to_copy = [
        ("docs/PILOT_OFFER.md", "PILOT_OFFER.md"),
        ("docs/PILOT_SOW.md", "PILOT_SOW.md"),
        ("docs/PILOT_README.md", "PILOT_PLAYBOOK.md"),
    ]
    for src_rel, dst_name in docs_to_copy:
        copy_if_exists(REPO_ROOT / src_rel, outreach / dst_name)

    tpl = REPO_ROOT / "templates/pilot/outreach_one_pager.md"
    if tpl.exists():
        ctx = build_template_context(cfg)
        (outreach / "ONE_PAGER.md").write_text(render_template(tpl, ctx))

    index = {
        "pilot_id": cfg["pilot"]["id"],
        "outreach_dir": as_repo_relative(outreach),
        "files": sorted(p.name for p in outreach.iterdir() if p.is_file()),
        "checklist_before_outreach": [
            "Fill contact_email and company_name in pilot config",
            "Run --full-meep if sim_budget_curve only has B=15,30",
            "Review SIM_CONTRACT exclusions with legal if needed",
            "Attach design_dossier + ONE_PAGER to first email",
        ],
    }
    (outreach / "README.md").write_text(
        "# Pilot outreach bundle\n\n"
        f"**Pilot:** `{cfg['pilot']['id']}`\n\n"
        "## Attach to first email\n\n"
        "- `ONE_PAGER.md` (or export PDF)\n"
        "- `SIM_BUDGET_REPORT.md` + `sim_budget_curve.png`\n"
        "- `design_dossier.csv` + sample PNGs from `designs/`\n\n"
        "## After interest\n\n"
        "- `PILOT_OFFER.md` — scope & pricing placeholders\n"
        "- `PILOT_SOW.md` — IP & data terms\n"
        "- `SIM_CONTRACT.md` — what MEEP qualification means\n\n"
        "## Internal\n\n"
        "- `PILOT_PLAYBOOK.md` — how to run the pipeline\n"
    )
    (outreach / "index.json").write_text(json.dumps(index, indent=2))
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
