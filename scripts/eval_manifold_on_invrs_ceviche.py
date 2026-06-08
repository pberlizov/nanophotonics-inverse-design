#!/usr/bin/env python3
"""Evaluate drcgenerator champion masks on invrs-gym Ceviche (Track C / D2 bridge).

Does not use MEEP. Maps 4 µm EBL masks → 1.6 µm Ceviche density grid.

  PYTHONPATH=src python scripts/eval_manifold_on_invrs_ceviche.py
  PYTHONPATH=src python scripts/eval_manifold_on_invrs_ceviche.py --soft
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nano_inv.invrs_adapter import (
    ChallengeName,
    evaluate_gym_params,
    mask_to_gym_params,
)

DEFAULT_CFG = REPO / "configs" / "invrs_manifold_eval.yaml"
OUT = REPO / "data" / "phase1" / "invrs_benchmark"


def load_cfg(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--challenge", default=None)
    p.add_argument("--soft", action="store_true", help="Use decode_soft density")
    p.add_argument("--sample-id", default=None)
    args = p.parse_args()

    cfg = load_cfg(args.config)
    challenge: ChallengeName = args.challenge or cfg.get(
        "challenge", "ceviche_lightweight_power_splitter"
    )
    samples = cfg.get("samples") or []
    if args.sample_id:
        samples = [s for s in samples if s["id"] == args.sample_id]

    rows: list[dict] = []
    for s in samples:
        sid = s["id"]
        mask_path = REPO / s["mask"]
        if not mask_path.exists():
            print(f"SKIP {sid}: missing {mask_path}")
            continue
        mask = np.load(mask_path)
        print(f"==> {sid} ({'soft' if args.soft else 'hard'})")
        params = mask_to_gym_params(
            mask,
            challenge=challenge,
            use_soft=args.soft,
        )
        ev = evaluate_gym_params(params, challenge=challenge)
        row = {
            "sample_id": sid,
            "mask_path": str(mask_path.relative_to(REPO)),
            "challenge": challenge,
            "density_mode": "soft" if args.soft else "hard",
            **ev,
        }
        rows.append(row)
        print(f"    eval_metric={ev['eval_metric']:.4f}  in_spec={ev['in_spec']}")

    OUT.mkdir(parents=True, exist_ok=True)
    tag = "soft" if args.soft else "hard"
    out_json = OUT / f"manifold_ceviche_{tag}.json"
    out_json.write_text(json.dumps(rows, indent=2))

    lines = [
        "# drcgenerator masks on invrs-gym Ceviche",
        "",
        f"Challenge: `{challenge}` · density: **{tag}**",
        "",
        "| sample | eval_metric | in_spec | loss |",
        "|--------|-------------|---------|------|",
    ]
    for r in rows:
        flag = "✓" if r["in_spec"] else "—"
        lines.append(
            f"| {r['sample_id']} | {r['eval_metric']:.4f} | {flag} | {r['loss']:.2f} |"
        )
    out_md = OUT / f"manifold_ceviche_{tag}.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_json}\nWrote {out_md}")


if __name__ == "__main__":
    main()
