#!/usr/bin/env python3
"""
Wedge A acquisition round: propose (cheap) → MEEP verify k → merge labels → retrain → ranking gate.

  python scripts/run_wedge_a_round.py --round 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.manifest import append_sim_results  # noqa: E402


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/wedge_a.yaml")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--skip-meep", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    rd = cfg["round"]
    sur = cfg["surrogate"]
    py = REPO_ROOT / ".venv/bin/python"
    meep_sh = REPO_ROOT / "scripts/run_meep.sh"

    round_dir = REPO_ROOT / cfg["data"]["wedge_root"] / f"rounds/round_{args.round:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    merge_into = REPO_ROOT / rd["merge_into"]
    n_prop = int(rd["n_proposals"])
    meep_k = int(rd["meep_verify_k"])

    cand_all = round_dir / "candidates.csv"
    run(
        [
            str(py),
            "scripts/generate_ranked_candidates.py",
            "--config",
            str(args.config),
            "--surrogate",
            sur["output_dir"],
            "--n-proposals",
            str(n_prop),
            "--output",
            str(cand_all.relative_to(REPO_ROOT)),
        ]
    )

    top = pd.read_csv(cand_all).head(meep_k)
    meep_manifest = round_dir / "meep_verify.csv"
    top.to_csv(meep_manifest, index=False)

    if not args.skip_meep:
        meep_out = round_dir / "meep_verify_results.csv"
        run(
            [
                "bash",
                str(meep_sh),
                "scripts/run_fdtd_batch.py",
                "--config",
                str(REPO_ROOT / "configs/phase0.yaml"),
                "--manifest",
                str(meep_manifest.relative_to(REPO_ROOT)),
                "--output",
                str(meep_out.relative_to(REPO_ROOT)),
                "--force-resim",
                "--no-skip-existing",
            ]
        )
        append_sim_results(merge_into, pd.read_csv(meep_out), out_path=merge_into)

    if not args.skip_train:
        run([str(py), "scripts/train_wedge_a_surrogate.py", "--config", str(args.config)])

    summary = {
        "round": args.round,
        "round_dir": str(round_dir.relative_to(REPO_ROOT)),
        "n_proposals": n_prop,
        "meep_verify_k": meep_k,
        "merge_into": str(merge_into.relative_to(REPO_ROOT)),
    }
    (round_dir / "round_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
