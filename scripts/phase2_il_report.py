#!/usr/bin/env python3
"""Release report for Phase 2 IL hunt."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data/phase1/phase2_il_hunt"
RELEASE = REPO / "data/phase1/release"


def main() -> None:
    s1 = pd.read_csv(OUT_DIR / "stage1_trials.csv")
    s2 = pd.read_csv(OUT_DIR / "stage2_trials.csv")
    summary = json.loads((OUT_DIR / "phase2_il_summary.json").read_text())

    best_il = s2.sort_values("IL_db").iloc[0]
    best_in_spec = s2[s2["in_spec"] == True].sort_values("IL_db")
    if len(best_in_spec) == 0:
        best_in_spec = s1[s1["in_spec"] == True].sort_values("IL_db").iloc[0]
    else:
        best_in_spec = best_in_spec.iloc[0]

    la = json.loads((RELEASE / "loss_aware_hunt.json").read_text()) if (RELEASE / "loss_aware_hunt.json").exists() else {}

    lines = [
        "# Phase 2 IL hunt (weight_il=0.75)",
        "",
        f"**Stage 1:** {summary['stage1_trials']} trials, {summary['stage1_in_spec']} in-spec split",
        f"**Stage 2:** {summary['stage2_trials']} trials (5 seeds × 30 refine)",
        f"**Best IL (overall):** {summary['best_IL_db']:.1f} dB (`{best_il['sample_id']}`, split err {best_il['split_err']:.4f})",
        f"**Best IL in-spec split:** {best_in_spec['IL_db']:.1f} dB (`{best_in_spec['sample_id']}`, split err {best_in_spec['split_err']:.4f})",
        "",
        "## vs. Phase B (loss-aware, weight_il=0.15)",
        "",
        "| design | split err | IL (dB) | in-spec |",
        "|--------|-----------|---------|---------|",
    ]
    if la:
        lines.append(
            f"| loss-aware best in-spec | — | {la.get('best_IL_db_in_spec', 0):.1f} | yes |"
        )
        bl = la.get("baselines", {}).get("cand_000261", {})
        lines.append(f"| `cand_000261` baseline | {bl.get('split_err', 0):.4f} | {bl.get('IL_db', 0):.1f} | yes |")
    lines.extend(
        [
            f"| Phase 2 best IL | {best_il['split_err']:.4f} | {best_il['IL_db']:.1f} | {bool(best_il['in_spec'])} |",
            f"| Phase 2 best in-spec | {best_in_spec['split_err']:.4f} | {best_in_spec['IL_db']:.1f} | yes |",
            "",
            "**Interpretation:** Stronger IL weighting drives simulated IL much lower, at the cost of split accuracy on the best-IL point. Use as a diagnostic tradeoff, not a promoted product splitter.",
            "",
            f"Artifacts: `{OUT_DIR.relative_to(REPO)}/`",
        ]
    )

    md = "\n".join(lines) + "\n"
    (RELEASE / "phase2_il_hunt.md").write_text(md)
    (RELEASE / "phase2_il_hunt.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {RELEASE / 'phase2_il_hunt.md'}")


if __name__ == "__main__":
    main()
