#!/usr/bin/env python3
"""Summarize loss-aware hunt → release/loss_aware_hunt.{md,json}."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
HUNT_DIR = REPO / "data/phase1/loss_aware_hunt"
RELEASE = REPO / "data/phase1/release"
CFG = REPO / "configs/loss_aware_hunt.yaml"


def main() -> None:
    csv_path = HUNT_DIR / "loss_aware_trials.csv"
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path} — run loss_aware_search.py first.")

    cfg = yaml.safe_load(CFG.read_text()) if CFG.exists() else {}
    tol = float(cfg.get("targets", {}).get("split_ratio_tolerance", 0.05))
    df = pd.read_csv(csv_path)
    df["in_spec"] = df["split_err"] <= tol

    # Champion baselines from FOM table for comparison
    baselines = {
        "cand_000261": {"IL_db": 19.4, "split_err": 0.0054},
        "local_00022": {"IL_db": 17.8, "split_err": 0.001},
    }

    in_spec = df[df["in_spec"]]
    best_il_in_spec = in_spec.sort_values("IL_db").head(1) if len(in_spec) else pd.DataFrame()
    best_split = df.sort_values("split_err").head(1)

    summary = {
        "n_trials": int(len(df)),
        "n_in_spec_split": int(in_spec.shape[0]),
        "best_split_err": float(df["split_err"].min()),
        "best_IL_db_overall": float(df["IL_db"].min()),
        "best_IL_db_in_spec": float(best_il_in_spec["IL_db"].iloc[0]) if len(best_il_in_spec) else None,
        "il_weight": float(cfg.get("search", {}).get("weight_il", 0.15)),
        "max_il_db_soft": float(cfg.get("search", {}).get("max_insertion_loss_db", 12.0)),
        "baselines": baselines,
        "publication_note": (
            "Phase B complete — update manuscript loss-aware subsection."
            if len(df) > 0
            else "No trials."
        ),
    }
    RELEASE.mkdir(parents=True, exist_ok=True)
    (RELEASE / "loss_aware_hunt.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Loss-aware hunt (split + IL objective)",
        "",
        f"**Trials:** {len(df)} | **In-spec split:** {summary['n_in_spec_split']}/{len(df)}",
        f"**Objective:** multi (`weight_il={summary['il_weight']}`, soft cap {summary['max_il_db_soft']} dB)",
        "",
        "## vs. split-only champions",
        "",
        "| design | split err | IL (dB) |",
        "|--------|-----------|---------|",
    ]
    for sid, b in baselines.items():
        lines.append(f"| `{sid}` (baseline) | {b['split_err']:.4f} | {b['IL_db']:.1f} |")
    if len(best_il_in_spec):
        r = best_il_in_spec.iloc[0]
        lines.append(
            f"| `{r['sample_id']}` (best IL in-spec) | {r['split_err']:.4f} | {r['IL_db']:.1f} |"
        )
    if len(best_split):
        r = best_split.iloc[0]
        lines.append(f"| `{r['sample_id']}` (best split) | {r['split_err']:.4f} | {r['IL_db']:.1f} |")

    lines.extend(["", "## Top 10 by composite loss", ""])
    top = df.sort_values("loss").head(10)
    cols = [c for c in ("sample_id", "center_id", "split_err", "IL_db", "loss", "in_spec") if c in top.columns]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for _, r in top.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    (RELEASE / "loss_aware_hunt.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
