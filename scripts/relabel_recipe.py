#!/usr/bin/env python3
"""Re-label manifest under a new MEEP recipe_version (separate CSV)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe-version", type=str, default="phase0_v1")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sample-ids", type=str, default=None)
    p.add_argument("--sources", type=str, default="perturb")
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--force-resim", action="store_true", default=True)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    out = REPO_ROOT / "data" / "phase0" / f"sim_results_{args.recipe_version}.csv"
    cmd = [
        "bash",
        str(REPO_ROOT / "scripts" / "run_meep.sh"),
        "scripts/run_fdtd_batch.py",
        "--recipe-version",
        args.recipe_version,
        "--output",
        str(out.relative_to(REPO_ROOT)),
        "--sources-filter",
        args.sources,
        "--no-skip-existing",
    ]
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    if args.sample_ids:
        cmd.extend(["--sample-ids", args.sample_ids])
    if args.resolution:
        cmd.extend(["--resolution", str(args.resolution)])
    if args.force_resim:
        cmd.append("--force-resim")
    if args.verbose:
        cmd.append("--verbose")

    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
