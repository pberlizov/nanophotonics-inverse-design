#!/usr/bin/env python3
"""Free disk: drop latent/mask caches after MEEP results exist (keeps CSV + meep_results)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CACHE_DIR_NAMES = ("latents", "masks", "phase_sigma", "phase_rank")
CANDIDATE_SUB = "candidates"


def dir_has_complete_meep(run_dir: Path, budget: int) -> bool:
    csv_path = run_dir / "meep_results.csv"
    if not csv_path.exists():
        return False
    try:
        import pandas as pd

        return len(pd.read_csv(csv_path)) >= budget
    except Exception:
        return False


def parse_budget_from_name(name: str) -> int | None:
    if "_B" not in name:
        return None
    try:
        return int(name.rsplit("_B", 1)[-1])
    except ValueError:
        return None


def cleanup_policy_dir(policy_dir: Path, *, dry_run: bool) -> int:
    freed = 0
    budget = parse_budget_from_name(policy_dir.name)
    if budget is None or not dir_has_complete_meep(policy_dir, budget):
        return 0
    for sub in CACHE_DIR_NAMES:
        p = policy_dir / sub
        if p.is_dir():
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            if dry_run:
                print(f"  would remove {p.relative_to(REPO_ROOT)} ({size / 1e6:.1f} MB)")
            else:
                shutil.rmtree(p)
            freed += size
    cand = policy_dir / CANDIDATE_SUB
    if cand.is_dir():
        size = sum(f.stat().st_size for f in cand.rglob("*") if f.is_file())
        if dry_run:
            print(f"  would remove {cand.relative_to(REPO_ROOT)} ({size / 1e6:.1f} MB)")
        else:
            shutil.rmtree(cand)
        freed += size
    return freed


def cleanup_replicate_run(run_dir: Path, *, dry_run: bool, require_report: bool) -> int:
    if require_report and not (run_dir / "run_report.json").exists():
        print(f"skip {run_dir.name} (no run_report.json)")
        return 0
    freed = 0
    for sub in CACHE_DIR_NAMES:
        p = run_dir / sub
        if p.is_dir():
            size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            if dry_run:
                print(f"  would remove {p.relative_to(REPO_ROOT)} ({size / 1e6:.1f} MB)")
            else:
                shutil.rmtree(p)
            freed += size
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in CACHE_DIR_NAMES or child.name == CANDIDATE_SUB:
            continue
        if "_B" in child.name:
            freed += cleanup_policy_dir(child, dry_run=dry_run)
    return freed


def cleanup_partial_hier(run_dir: Path, *, dry_run: bool) -> int:
    """Remove failed/incomplete hierarchical_* dirs (no meep_results.csv)."""
    freed = 0
    for child in run_dir.glob("hierarchical_*_B*"):
        if not child.is_dir() or (child / "meep_results.csv").exists():
            continue
        size = sum(f.stat().st_size for f in child.rglob("*") if f.is_file())
        if dry_run:
            print(f"  would remove partial {child.relative_to(REPO_ROOT)} ({size / 1e6:.1f} MB)")
        else:
            shutil.rmtree(child)
        freed += size
    return freed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--replicates-root",
        type=Path,
        default=REPO_ROOT / "data/phase1/wedge_a/sim_budget/replicates",
    )
    p.add_argument("--run", type=str, default=None, help="Only run_XX (e.g. 01 or run_01)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Also clean per-policy caches when meep_results.csv is complete (no run_report required)",
    )
    args = p.parse_args()
    root = args.replicates_root if args.replicates_root.is_absolute() else REPO_ROOT / args.replicates_root

    total = 0
    runs = sorted(root.glob("run_*"))
    if args.run:
        tag = args.run if args.run.startswith("run_") else f"run_{args.run}"
        runs = [root / tag]

    for run_dir in runs:
        if not run_dir.is_dir():
            continue
        print(f"==> {run_dir.name}")
        total += cleanup_partial_hier(run_dir, dry_run=args.dry_run)
        require_report = not args.include_incomplete
        total += cleanup_replicate_run(run_dir, dry_run=args.dry_run, require_report=require_report)

    print(f"\n{'would free' if args.dry_run else 'freed'} ~{total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
