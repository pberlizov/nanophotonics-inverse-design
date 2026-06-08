#!/usr/bin/env python3
"""
One active-learning round: BO (surrogate) -> MEEP verify top-k -> retrain improved surrogate.

Runs in .venv (drcgenerator + surrogate). MEEP step shells out to scripts/run_meep.sh.
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

from nano_inv.manifest import build_al_training_manifest  # noqa: E402


def resolve_repo_path(path: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase0.yaml")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--meep-top-k", type=int, default=None)
    p.add_argument("--search-top-k", type=int, default=20)
    p.add_argument(
        "--architecture",
        type=str,
        choices=["latent_mlp", "mask_mlp", "mask_cnn"],
        default=None,
        help="Surrogate to train after round",
    )
    p.add_argument("--sources", type=str, default=None)
    p.add_argument("--search-mode", type=str, default="perturb", choices=["perturb", "perlin", "mixed"])
    p.add_argument("--skip-search", action="store_true")
    p.add_argument("--skip-meep", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--surrogate-in", type=Path, default=None, help="For BO step")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def run(cmd: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = REPO_ROOT / cfg["data"]["root"]
    al_cfg = cfg.get("active_learning") or {}
    round_dir = data_root / f"al_round_{args.round:02d}"
    search_dir = round_dir / "search"
    surrogate_in = resolve_repo_path(
        args.surrogate_in or (data_root / "surrogate" / "surrogate.joblib")
    )
    surrogate_out = round_dir / "surrogate"

    n_trials = args.n_trials or int(al_cfg.get("n_trials", 100))
    meep_k = args.meep_top_k or int(al_cfg.get("meep_verify_top_k", 5))
    arch = args.architecture or al_cfg.get("architecture", "mask_mlp")
    sources = args.sources or al_cfg.get("source_filter", "perturb")

    py = REPO_ROOT / ".venv/bin/python"
    meep_sh = REPO_ROOT / "scripts/run_meep.sh"

    plan = {
        "round": args.round,
        "round_dir": str(round_dir.relative_to(REPO_ROOT)),
        "n_trials": n_trials,
        "meep_verify_top_k": meep_k,
        "retrain_architecture": arch,
        "retrain_sources": sources,
    }
    print(json.dumps(plan, indent=2))

    if not args.skip_search:
        cmd = [
            str(py),
            "scripts/latent_search.py",
            "--output-dir",
            str(search_dir),
            "--n-trials",
            str(n_trials),
            "--top-k",
            str(args.search_top_k),
            "--mode",
            args.search_mode,
            "--surrogate",
            str(surrogate_in.resolve()),
        ]
        run(cmd, dry_run=args.dry_run)

    cand_manifest = search_dir / "top_candidates.csv"
    if not args.skip_meep:
        # MEEP only top meep_k rows
        if not args.dry_run:
            cand = pd.read_csv(cand_manifest)
            top = cand.head(meep_k)
            meep_manifest = round_dir / "meep_verify.csv"
            top.to_csv(meep_manifest, index=False)
            cand_manifest = meep_manifest

        cmd = [
            "bash",
            str(meep_sh),
            "scripts/run_fdtd_batch.py",
            "--manifest",
            str(cand_manifest.relative_to(REPO_ROOT)),
            "--no-skip-existing",
            "--force-resim",
        ]
        run(cmd, dry_run=args.dry_run)

    manifest_train = round_dir / "manifest_train.csv"
    meep_manifest = round_dir / "meep_verify.csv"
    train_candidates = (
        meep_manifest if meep_manifest.exists() else search_dir / "top_candidates.csv"
    )
    if not args.skip_train and not args.dry_run:
        base_manifest = REPO_ROOT / cfg["data"]["manifest"]
        combined = build_al_training_manifest(
            REPO_ROOT,
            base_manifest,
            train_candidates,
            manifest_train,
            base_source_filter=sources if sources in ("perturb", "perlin") else "perturb",
        )
        plan["manifest_train"] = str(manifest_train.relative_to(REPO_ROOT))
        plan["n_manifest_train_rows"] = len(combined)
        print(f"Training manifest: {manifest_train} ({len(combined)} rows)")

    if not args.skip_train:
        cmd = [
            str(py),
            "scripts/train_surrogate.py",
            "--output-dir",
            str(surrogate_out),
            "--architecture",
            arch,
            "--sources",
            "all",
            "--manifest",
            str(manifest_train.relative_to(REPO_ROOT)),
        ]
        run(cmd, dry_run=args.dry_run)

    if not args.dry_run:
        eval_cmd = [str(py), "scripts/evaluate_phase0.py", "--output", str(round_dir / "gate_metrics.json")]
        run(eval_cmd, dry_run=False)

    summary_path = round_dir / "round_summary.json"
    if not args.dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(plan, indent=2))
        print(f"\nRound {args.round} done. Summary: {summary_path.relative_to(REPO_ROOT)}")
        print("Next: point --surrogate-in at", surrogate_out / "surrogate.joblib", "for round", args.round + 1)
    else:
        print("\n(dry-run — no files written)")


if __name__ == "__main__":
    main()
