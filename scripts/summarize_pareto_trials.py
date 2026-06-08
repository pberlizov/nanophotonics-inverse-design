#!/usr/bin/env python3
"""B4 — Extract non-dominated trials on (split error, IL) from a search trials CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.search_objectives import dominates, pareto_tuple  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=Path, required=True)
    p.add_argument("--target", type=float, default=0.5)
    p.add_argument("--max-il", type=float, default=12.0)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.trials if args.trials.is_absolute() else REPO_ROOT / args.trials)
    if "meep_split_ratio_upper" not in df.columns:
        raise SystemExit("trials CSV needs meep_split_ratio_upper column")

    points: list[tuple[int, tuple[float, float], dict]] = []
    for _, r in df.iterrows():
        if pd.isna(r.get("meep_split_ratio_upper")):
            continue
        il = float(r.get("insertion_loss_db", 20.0))
        se, ilp, _ = pareto_tuple(
            float(r["meep_split_ratio_upper"]),
            il,
            args.target,
            max_il_db=args.max_il,
        )
        points.append((int(r.get("trial_number", len(points))), (se, ilp), r.to_dict()))

    front: list[tuple[int, tuple[float, float], dict]] = []
    for tid, obj, row in points:
        if any(dominates(other[1], obj) for other in points if other[0] != tid):
            continue
        front.append((tid, obj, row))

    out = {
        "n_trials": len(df),
        "n_pareto": len(front),
        "pareto_trials": [
            {"trial": tid, "split_err": o[0], "il_penalty": o[1], "sample_id": row.get("sample_id")}
            for tid, o, row in sorted(front, key=lambda x: x[1][0])
        ],
    }
    text = json.dumps(out, indent=2)
    if args.output:
        path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
