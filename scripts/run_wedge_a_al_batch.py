#!/usr/bin/env python3
"""Run multiple wedge A AL rounds; optional retrain after each round."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.pilot import load_pilot_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/wedge_a_production.yaml")
    p.add_argument("--from-round", type=int, default=2)
    p.add_argument("--to-round", type=int, default=5)
    p.add_argument("--skip-meep", action="store_true")
    p.add_argument("--no-retrain-between", action="store_true")
    args = p.parse_args()

    cfg = load_pilot_config(args.config)
    al = cfg.get("al", {})
    round_cfg_path = REPO_ROOT / al.get("config", "configs/wedge_a_mask.yaml")

    # Apply AL overrides into round section of AL config file via env temp - simpler: patch round in merged cfg
    al_cfg = load_pilot_config(round_cfg_path)
    if al.get("meep_verify_k"):
        al_cfg.setdefault("round", {})["meep_verify_k"] = int(al["meep_verify_k"])
    if al.get("n_proposals"):
        al_cfg.setdefault("round", {})["n_proposals"] = int(al["n_proposals"])
    if al.get("merge_into"):
        al_cfg.setdefault("round", {})["merge_into"] = al["merge_into"]

    tmp_cfg = REPO_ROOT / "data/phase1/wedge_a/al_batch_config.yaml"
    tmp_cfg.parent.mkdir(parents=True, exist_ok=True)
    import yaml as _yaml

    tmp_cfg.write_text(_yaml.safe_dump(al_cfg))

    py = REPO_ROOT / ".venv/bin/python"
    retrain = not args.no_retrain_between and al.get("retrain_after_each_round", True)

    summaries = []
    for r in range(args.from_round, args.to_round + 1):
        cmd = [
            str(py),
            "scripts/run_wedge_a_round.py",
            "--config",
            str(tmp_cfg),
            "--round",
            str(r),
        ]
        if args.skip_meep:
            cmd.append("--skip-meep")
        if not retrain:
            cmd.append("--skip-train")
        print("+", " ".join(cmd))
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        summary_path = REPO_ROOT / al_cfg["data"]["wedge_root"] / f"rounds/round_{r:02d}" / "round_summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text()))

    (REPO_ROOT / "data/phase1/wedge_a/al_batch_summary.json").write_text(
        json.dumps({"rounds": summaries}, indent=2)
    )


if __name__ == "__main__":
    main()
