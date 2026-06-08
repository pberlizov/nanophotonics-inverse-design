#!/usr/bin/env python3
"""
B7 — Run comparable search baselines and record sim-budget vs best FOM.

Pilots use small n-trials; use --full for production-scale comparison.

  python scripts/run_baselines.py --pilot
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, dry: bool) -> None:
    print("+", " ".join(cmd))
    if not dry:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def load_summary(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/phase1_track_b.yaml")
    p.add_argument("--pilot", action="store_true", help="Small trial counts (~5 min total)")
    p.add_argument("--full", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    b7 = (cfg.get("track_b") or {}).get("baselines") or {}
    n_pilot = int(b7.get("n_trials_pilot", 15))
    n_full = int(b7.get("n_trials_full", 100))
    n = n_full if args.full else n_pilot

    out_dir = REPO_ROOT / b7.get("output_dir", "data/phase1/track_b/baselines")
    out_dir.mkdir(parents=True, exist_ok=True)
    meep = ["bash", "scripts/run_meep.sh"]

    baselines = [
        (
            "sigma_meep",
            meep
            + [
                "scripts/meep_search.py",
                "--config",
                "configs/phase1_track_b.yaml",
                "--n-trials",
                str(n),
                "--output-dir",
                str((out_dir / "sigma_meep").relative_to(REPO_ROOT)),
                "--objective",
                "multi",
            ],
        ),
        (
            "latent_residual_meep",
            meep
            + [
                "scripts/latent_meep_search.py",
                "--config",
                "configs/phase1_track_b.yaml",
                "--latent-mode",
                "residual",
                "--n-trials",
                str(n),
                "--output-dir",
                str((out_dir / "latent_residual_meep").relative_to(REPO_ROOT)),
            ],
        ),
        (
            "latent_pca_meep",
            meep
            + [
                "scripts/latent_meep_search.py",
                "--config",
                "configs/phase1_track_b.yaml",
                "--latent-mode",
                "pca",
                "--n-trials",
                str(max(n // 2, 10)),
                "--output-dir",
                str((out_dir / "latent_pca_meep").relative_to(REPO_ROOT)),
            ],
        ),
    ]

    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "n_trials_per_baseline": n,
        "baselines": {},
    }

    for name, cmd in baselines:
        if not args.dry_run:
            try:
                run(cmd, dry=False)
            except subprocess.CalledProcessError as exc:
                report["baselines"][name] = {"error": str(exc)}
                continue
        summary_paths = [
            out_dir / name / "meep_search_summary.json",
            out_dir / name / "latent_meep_summary.json",
        ]
        summ = None
        for sp in summary_paths:
            summ = load_summary(sp)
            if summ:
                break
        report["baselines"][name] = summ or {"status": "dry_run" if args.dry_run else "missing_summary"}

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "baselines_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
