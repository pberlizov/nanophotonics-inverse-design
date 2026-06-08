#!/usr/bin/env python3
"""
Multi-stage C-band broadband discovery for preprint contribution.

Stages (run separately or via --stage all):
  pool     — collect in-spec candidate masks from prior searches (CPU)
  rescore  — coarse broadband eval on pool (MEEP)
  refine   — residual Optuna around champion latents (MEEP)
  explore  — latent_meep_search broadband exploration (MEEP subprocess)
  verify   — fine-grid gate on top candidates (MEEP)
  report   — merge results → release/broadband_hunt.md (CPU)

Target contribution: designs with worst |R_up−0.5| ≤ 0.05 over 1.53–1.57 µm.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
VENV_PY = REPO / ".venv/bin/python"
OUT_RELEASE = REPO / "data/phase1/release"
sys.path.insert(0, str(REPO / "src"))

POOL_SOURCES = [
    "data/phase1/meep_search_local/top_candidates.csv",
    "data/phase1/meep_search_local_deep/top_candidates.csv",
    "data/phase1/meep_search/top_candidates.csv",
    "data/phase1/meep_search_deep/top_candidates.csv",
    "data/phase0/meep_search_100/top_candidates.csv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO / "configs/broadband_hunt.yaml")
    p.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["pool", "rescore", "refine", "explore", "verify", "report", "all"],
    )
    p.add_argument("--skip-explore", action="store_true", help="Skip latent explore (faster)")
    return p.parse_args()


def hunt_dir(cfg: dict) -> Path:
    d = REPO / (cfg.get("hunt") or {}).get("output_dir", "data/phase1/broadband_hunt")
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_pool(cfg: dict) -> Path:
    out = hunt_dir(cfg)
    rows = []
    for rel in POOL_SOURCES:
        p = REPO / rel
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "in_spec" in df.columns:
            df = df[df["in_spec"] == True]  # noqa: E712
        for _, r in df.iterrows():
            rows.append(
                {
                    "source_csv": rel,
                    "sample_id": r.get("sample_id"),
                    "split_ratio_upper": r.get("split_ratio_upper"),
                    "mask_path": r.get("mask_path"),
                    "latent_path": r.get("latent_path"),
                }
            )
    pool = pd.DataFrame(rows).drop_duplicates(subset=["sample_id"])
    pool_path = out / "candidate_pool.csv"
    pool.to_csv(pool_path, index=False)
    print(f"pool: {len(pool)} in-spec candidates → {pool_path}")
    return pool_path


def run_meep_script(script: str, *extra: str) -> None:
    cmd = ["bash", str(REPO / "scripts/run_meep.sh"), script, *extra]
    subprocess.run(cmd, cwd=REPO, check=True)


def stage_rescore(cfg: dict) -> None:
    pool = hunt_dir(cfg) / "candidate_pool.csv"
    if not pool.exists():
        stage_pool(cfg)
    bb = cfg.get("broadband") or {}
    out_json = hunt_dir(cfg) / "pool_broadband_rescore.json"
    extra = ["--resume"] if out_json.exists() else []
    run_meep_script(
        "scripts/broadband_rescore_candidates.py",
        "--candidates",
        str(pool.relative_to(REPO)),
        "--config",
        "configs/broadband_hunt.yaml",
        "--wl-step",
        str(bb.get("wl_step", 0.01)),
        "--max-worst-split-error",
        str(bb.get("max_worst_split_error", 0.05)),
        "--output",
        str(out_json.relative_to(REPO)),
        *extra,
    )


def stage_refine(cfg: dict) -> None:
    out = (cfg.get("hunt") or {}).get("output_dir", "data/phase1/broadband_hunt/refine")
    run_meep_script(
        "scripts/broadband_refine_from_centers.py",
        "--config",
        str((REPO / "configs/broadband_hunt.yaml").relative_to(REPO)),
        "--output-dir",
        str(out),
    )


def stage_explore(cfg: dict) -> None:
    hunt = cfg.get("hunt") or {}
    n = int(hunt.get("latent_explore_trials", 40))
    out = str(hunt_dir(cfg) / "explore")
    run_meep_script(
        "scripts/latent_meep_search.py",
        "--config",
        str((REPO / "configs/broadband_hunt.yaml").relative_to(REPO)),
        "--output-dir",
        out,
        "--objective",
        "broadband",
        "--latent-mode",
        str(hunt.get("latent_mode", "residual")),
        "--n-trials",
        str(n),
    )


def stage_verify(cfg: dict) -> None:
    """Fine-grid verify on best candidates from refine + explore + pool rescore."""
    hunt = hunt_dir(cfg)
    candidates: list[dict] = []

    refine_top = hunt / "refine" / "broadband_refine_top.csv"
    if refine_top.exists():
        for _, r in pd.read_csv(refine_top).head(5).iterrows():
            candidates.append({"sample_id": r["sample_id"], "mask_path": r["mask_path"]})

    explore_top = hunt / "explore" / "top_candidates.csv"
    if explore_top.exists():
        for _, r in pd.read_csv(explore_top).head(5).iterrows():
            candidates.append({"sample_id": r["sample_id"], "mask_path": r["mask_path"]})

    pool_json = hunt / "pool_broadband_rescore.json"
    if pool_json.exists():
        data = json.loads(pool_json.read_text())
        for r in sorted(data.get("results", []), key=lambda x: x.get("worst_split_error", 99))[:5]:
            if r.get("status") == "ok":
                candidates.append({"sample_id": r["sample_id"], "mask_path": r["mask_path"]})

    if not candidates:
        print("verify: no candidates to check")
        return

    verify_csv = hunt / "verify_candidates.csv"
    pd.DataFrame(candidates).drop_duplicates("sample_id").to_csv(verify_csv, index=False)
    bb = cfg.get("broadband") or {}
    run_meep_script(
        "scripts/broadband_rescore_candidates.py",
        "--candidates",
        str(verify_csv.relative_to(REPO)),
        "--wl-step",
        str(bb.get("verify_wl_step", 0.005)),
        "--max-worst-split-error",
        str(bb.get("max_worst_split_error", 0.05)),
        "--output",
        str((hunt / "broadband_verify.json").relative_to(REPO)),
    )


def stage_report(cfg: dict) -> None:
    hunt = hunt_dir(cfg)
    bb = cfg.get("broadband") or {}
    gate = float(bb.get("max_worst_split_error", 0.05))

    sections: list[dict] = []
    for name, path in [
        ("pool_rescore", hunt / "pool_broadband_rescore.json"),
        ("refine", hunt / "refine" / "broadband_refine_summary.json"),
        ("verify", hunt / "broadband_verify.json"),
    ]:
        if path.exists():
            sections.append({"name": name, "data": json.loads(path.read_text())})

    winners: list[dict] = []
    verify_path = hunt / "broadband_verify.json"
    if verify_path.exists():
        for r in json.loads(verify_path.read_text()).get("results", []):
            if r.get("pass_broadband_gate"):
                winners.append(r)

    # Also promote any pool passers at coarse grid
    pool_path = hunt / "pool_broadband_rescore.json"
    if pool_path.exists():
        for r in json.loads(pool_path.read_text()).get("results", []):
            if r.get("pass_broadband_gate") and r["sample_id"] not in {w["sample_id"] for w in winners}:
                winners.append({**r, "note": "coarse_grid_only"})

    payload = {
        "gate_max_worst_split_error": gate,
        "wavelength_band_um": [bb.get("wl_start"), bb.get("wl_stop")],
        "n_winners": len(winners),
        "winners": winners,
        "sections": sections,
    }
    OUT_RELEASE.mkdir(parents=True, exist_ok=True)
    out_json = OUT_RELEASE / "broadband_hunt.json"
    out_json.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Broadband hunt — contribution track\n",
        f"**Gate:** worst |R_up − 0.5| ≤ {gate:.2f} over {bb.get('wl_start')}–{bb.get('wl_stop')} µm\n",
        f"**Verified winners:** {len(winners)}\n\n",
    ]
    if winners:
        lines.append("## Broadband-flat candidates\n\n")
        lines.append("| sample_id | worst |err| | mask_path | note |\n")
        lines.append("|-----------|-------------|-----------|------|\n")
        for w in sorted(winners, key=lambda x: x.get("worst_split_error", 99)):
            lines.append(
                f"| {w['sample_id']} | {w['worst_split_error']:.4f} | `{w.get('mask_path','')}` | "
                f"{w.get('note', 'fine verify')} |\n"
            )
    else:
        lines.append(
            "No candidates pass the fine-grid gate yet. "
            "Run `bash scripts/run_broadband_hunt.sh` (MEEP, multi-hour) or increase trials.\n"
        )

    lines.append("\n## Narrowband baseline (current champions)\n\n")
    champ_json = OUT_RELEASE / "champion_broadband.json"
    if champ_json.exists():
        for r in json.loads(champ_json.read_text()):
            lines.append(
                f"- `{r['sample_id']}`: worst |err| = {r['worst_split_error']:.3f} "
                f"(narrowband-tuned, fails C-band gate)\n"
            )

    lines.append("\n## Paper hook\n\n")
    lines.append(
        "Contrast **single-λ inverse design** (50/50 @ 1550 nm, dispersive off-band) with "
        "**broadband-aware latent refinement** (worst-λ + flatness objective). "
        "See `docs/preprint/BROADBAND_CONTRIBUTION.md`.\n"
    )

    out_md = OUT_RELEASE / "broadband_hunt.md"
    out_md.write_text("".join(lines))
    print(f"wrote {out_json}")
    print(f"wrote {out_md} ({len(winners)} winners)")


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    stages = (
        ["pool", "rescore", "refine", "explore", "verify", "report"]
        if args.stage == "all"
        else [args.stage]
    )

    for st in stages:
        print(f"\n===== stage: {st} =====")
        if st == "pool":
            stage_pool(cfg)
        elif st == "rescore":
            stage_rescore(cfg)
        elif st == "refine":
            stage_refine(cfg)
        elif st == "explore":
            if not args.skip_explore:
                stage_explore(cfg)
            else:
                print("skip explore")
        elif st == "verify":
            stage_verify(cfg)
        elif st == "report":
            stage_report(cfg)


if __name__ == "__main__":
    main()
