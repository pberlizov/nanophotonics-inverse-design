#!/usr/bin/env python3
"""Write reproducibility manifest for preprint release."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/phase1/release/repro_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def pkg_version(name: str) -> str:
    try:
        import importlib.metadata as im

        return im.version(name)
    except Exception:
        return ""


def main() -> None:
    corpus = REPO / "data/phase0/sim_results_phase0_v1_all.csv"
    masks = [
        REPO / "data/phase0/masks/ref_published_mask.npy",
        REPO / "data/phase1/wedge_a/meep_gated_shortlist/round_rank/masks/cand_000261_mask.npy",
    ]
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(run(["git", "status", "--porcelain"])),
        "python": sys.version.split()[0],
        "meep": run([str(REPO / ".venv/bin/python"), "-c", "import meep; print(meep.__version__)"]),
        "packages": {
            "numpy": pkg_version("numpy"),
            "pandas": pkg_version("pandas"),
            "scikit-learn": pkg_version("scikit-learn"),
            "torch": pkg_version("torch"),
        },
        "frozen_recipes": [
            "configs/phase0.yaml",
            "configs/promote_sdf_geom.yaml",
            "configs/wedge_a_production.yaml",
        ],
        "surrogate_weights": "data/phase1/wedge_a/surrogate_improved/surrogate.joblib",
        "corpus_sha256": sha256(corpus) if corpus.exists() else None,
        "champion_mask_sha256": {str(p.relative_to(REPO)): sha256(p) for p in masks if p.exists()},
        "sim_budget_seeds": "base_seed=2026, replicate seed = base + r*1000 for r=1..N",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
