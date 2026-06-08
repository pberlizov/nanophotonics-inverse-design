#!/usr/bin/env python3
"""B2 — Train split surrogates: perturb-only, perlin-only, perturb latent_mlp."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_train(cmd: list[str]) -> dict:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    metrics_path = Path(cmd[cmd.index("--output-dir") + 1]) / "train_summary.json"
    return json.loads(metrics_path.read_text())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/phase1_track_b.yaml")
    args = p.parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    tb = cfg.get("track_b") or {}
    b2 = tb.get("surrogates") or {}
    sim = REPO_ROOT / cfg["data"].get("sim_corpus_v1", "data/phase0/sim_results_phase0_v1_all.csv")
    manifest = REPO_ROOT / cfg["data"]["manifest"]
    py = REPO_ROOT / ".venv/bin/python"
    out_root = REPO_ROOT / b2.get("output_root", "data/phase1/track_b/surrogates")

    jobs = [
        ("perturb_mask_mlp", "mask_mlp", "perturb"),
        ("perturb_latent_mlp", "latent_mlp", "perturb"),
        ("perlin_mask_mlp", "mask_mlp", "perlin"),
        ("all_mask_mlp", "mask_mlp", "all"),
    ]

    summary = {}
    for name, arch, sources in jobs:
        out = out_root / name
        metrics = run_train(
            [
                str(py),
                "scripts/train_surrogate.py",
                "--sim-results",
                str(sim.relative_to(REPO_ROOT)),
                "--manifest",
                str(manifest.relative_to(REPO_ROOT)),
                "--architecture",
                arch,
                "--sources",
                sources,
                "--output-dir",
                str(out.relative_to(REPO_ROOT)),
                "--min-ok",
                str(int(b2.get("min_ok", 50))),
            ]
        )
        summary[name] = metrics

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "train_track_b_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
