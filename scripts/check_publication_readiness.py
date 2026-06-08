#!/usr/bin/env python3
"""Gate check for single arXiv upload (v1 engineering + IL phase)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / "data/phase1/release"


def main() -> int:
    # Reuse v1 checker
    subprocess.run([sys.executable, str(REPO / "scripts/check_preprint_v1_readiness.py")], check=False)

    print()
    print("=" * 60)
    print("Publication gates (Phase B — IL objective)")
    print("=" * 60)

    blocking = True
    la_json = RELEASE / "loss_aware_hunt.json"
    la_csv = REPO / "data/phase1/loss_aware_hunt/loss_aware_trials.csv"

    if la_csv.exists() and la_csv.stat().st_size > 100:
        print("  [B] OK      loss_aware_trials.csv")
    else:
        print("  [B] MISSING loss_aware_trials.csv  ← run loss_aware_search.py")
        blocking = False  # not started yet

    if la_json.exists():
        data = json.loads(la_json.read_text())
        n = data.get("n_trials", 0)
        n_spec = data.get("n_in_spec_split", 0)
        best_il = data.get("best_IL_db_in_spec")
        print(f"  [B] OK      loss_aware_hunt.json  ({n} trials, {n_spec} in-spec split)")
        if best_il is not None:
            print(f"      Best IL among in-spec: {best_il:.1f} dB (baseline cand_000261: 19.4 dB)")
    else:
        print("  [B] MISSING loss_aware_hunt.json  ← run loss_aware_report.py")
        blocking = False

    print("-" * 60)
    if not la_csv.exists():
        print("NOT READY for arXiv: Phase B (IL objective) not run.")
        print("  bash scripts/run_meep.sh scripts/loss_aware_search.py")
        return 2
    if not la_json.exists():
        print("ALMOST: run python scripts/loss_aware_report.py then finalize.")
        return 1
    print("READY for manuscript update + arXiv (review loss-aware results in loss_aware_hunt.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
