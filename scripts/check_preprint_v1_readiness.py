#!/usr/bin/env python3
"""Print v1 preprint readiness against docs/preprint/V1_BACKLOG.md gates."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / "data/phase1/release"
FIGURES = REPO / "docs/preprint/figures"
MANUSCRIPT = REPO / "docs/preprint/manuscript.pdf"
REPLICATES = REPO / "data/phase1/wedge_a/sim_budget/replicates"


def ok(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def count_replicates() -> int:
    if not REPLICATES.exists():
        return 0
    return len([p for p in REPLICATES.iterdir() if p.is_dir() and p.name.startswith("run_")])


def main() -> int:
    checks: list[tuple[str, str, bool, str]] = []

    def add(tier: str, name: str, passed: bool, note: str = "") -> None:
        checks.append((tier, name, passed, note))

    # Tier A artifacts
    for stem in (
        "champion_fom_table",
        "flux_il_audit",
        "champion_broadband",
        "champion_mesh_convergence",
        "repro_manifest",
        "surrogate_validation",
        "ablation_proposal_pool",
        "novelty_extended",
        "champion_fab_stress",
        "broadband_hunt",
    ):
        add("A", stem, ok(RELEASE / f"{stem}.json") or ok(RELEASE / f"{stem}.md"))

    n_rep = count_replicates()
    add("A", "sim_budget_replicates", n_rep >= 6, f"{n_rep} run_* dirs (target 20 for final)")
    add("A", "sim_budget_stats", ok(REPO / "data/phase1/wedge_a/sim_budget_replication_stats.csv"))

    bb_winners = 0
    bb_path = RELEASE / "broadband_hunt.json"
    if bb_path.exists():
        try:
            bb_winners = int(json.loads(bb_path.read_text()).get("n_verified_winners", 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    add("A", "broadband_winners", True, f"{bb_winners} verified (0 OK for v1 — document limitation)")

    morph_summary = REPO / "data/phase1/morph_robust_hunt/morph_robust_summary.json"
    add("B", "morph_robust_hunt", ok(morph_summary), "in progress" if not morph_summary.exists() else "done")

    # Tier C figures
    for fig in (
        "pipeline_schematic.pdf",
        "hamming_cdf.pdf",
        "sim_budget_curve.pdf",
        "champion_mesh_convergence.pdf",
        "champion_fab_stress.pdf",
        "broadband_contribution.pdf",
        "loss_aware_tradeoff.pdf",
    ):
        add("C", fig, ok(FIGURES / fig))

    add("C", "manuscript.pdf", ok(MANUSCRIPT))

    # Blocking for v1: core artifacts + PDF; replication n=20 is soft
    blocking = {
        "champion_fom_table",
        "flux_il_audit",
        "champion_broadband",
        "champion_mesh_convergence",
        "repro_manifest",
        "manuscript.pdf",
    }

    print("=" * 60)
    print("Preprint v1 readiness")
    print("=" * 60)
    for tier, name, passed, note in checks:
        mark = "OK" if passed else "MISSING"
        extra = f"  ({note})" if note else ""
        print(f"  [{tier}] {mark:7} {name}{extra}")

    n_block_fail = sum(1 for _, n, p, _ in checks if n in blocking and not p)
    print("-" * 60)
    print(f"Replicates on disk: {n_rep} (v1 can ship with 6-seed pilot if labeled)")
    print(f"Broadband verified winners: {bb_winners}")

    if n_block_fail:
        print(f"\nBLOCKING: {n_block_fail} required item(s) missing.")
        return 1

    if n_rep < 20:
        print("\nREADY for v1 as PRELIMINARY (six-seed budget; n=20 in progress).")
    else:
        print("\nREADY for v1 (update PRIMARY_REPLICATES + budget table for n=20).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
