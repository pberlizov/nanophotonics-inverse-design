#!/usr/bin/env python3
"""Evaluate trained surrogate variants and write comparison CSV (no retrain)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/surrogate_improvement.yaml")
    p.add_argument(
        "--train-only",
        action="store_true",
        help="Deprecated no-op (training is step 2 of run_surrogate_improvement.sh)",
    )
    args = p.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    py = REPO_ROOT / ".venv/bin/python"
    rows = []
    for var in cfg["variants"]:
        out_dir = REPO_ROOT / var.get("output_dir", var["name"])
        metrics_path = out_dir / "metrics.json"
        if not metrics_path.exists():
            print(f"skip missing {metrics_path}")
            continue
        m = json.loads(metrics_path.read_text())
        eval_out = REPO_ROOT / "data/phase1" / f"ranking_eval_{var['name']}.json"
        subprocess.run(
            [
                str(py),
                "scripts/evaluate_surrogate_ranking.py",
                "--surrogate",
                str(out_dir.relative_to(REPO_ROOT)),
                "--sim-results",
                cfg["corpus"],
                "--sources",
                "perturb",
                "--output",
                str(eval_out.relative_to(REPO_ROOT)),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        r = json.loads(eval_out.read_text()) if eval_out.exists() else {}
        rows.append(
            {
                "name": var["name"],
                "architecture": m.get("architecture"),
                "target": m.get("target"),
                "val_r2": m.get("val_r2"),
                "val_mae": m.get("val_mae"),
                "n_ok": m.get("n_ok_total"),
                "ranking_wins": r.get("ranking_wins"),
                "mean_abs_err_topk": r.get("mean_abs_err_surrogate_topk"),
            }
        )

    df = pd.DataFrame(rows)
    out = REPO_ROOT / "data/phase1/surrogate_variant_comparison.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
