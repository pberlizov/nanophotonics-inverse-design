#!/usr/bin/env python3
"""
Surrogate-ranked active learning: presearch (cheap) -> MEEP verify top-k -> merge labels -> retrain.

Requires ranking_wins on prior eval unless --force.

  python scripts/surrogate_ranked_al_round.py --round 1
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

from nano_inv.manifest import append_sim_results, build_al_training_manifest  # noqa: E402


def run(cmd: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase1.yaml")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--force", action="store_true", help="Skip ranking_wins gate")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-presearch", action="store_true")
    p.add_argument("--skip-meep", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-ranking", action="store_true")
    return p.parse_args()


def check_ranking_gate(eval_path: Path) -> None:
    if not eval_path.exists():
        raise SystemExit(f"missing ranking eval: {eval_path} — run evaluate_surrogate_ranking.py")
    data = json.loads(eval_path.read_text())
    if not data.get("ranking_wins"):
        raise SystemExit(
            "ranking_wins is false — surrogate presearch is not validated. "
            "Use --force to override."
        )


def main() -> None:
    args = parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    pre_cfg = cfg.get("surrogate_presearch") or {}
    al_cfg = cfg.get("active_learning") or {}
    data_cfg = cfg.get("data") or {}

    if not args.force:
        eval_path = REPO_ROOT / pre_cfg.get("ranking_eval", "data/phase1/surrogate_ranking_eval.json")
        check_ranking_gate(eval_path)

    round_dir = REPO_ROOT / "data/phase1" / f"al_round_{args.round:02d}"
    pre_dir = round_dir / "presearch"
    py = REPO_ROOT / ".venv/bin/python"
    meep_sh = REPO_ROOT / "scripts/run_meep.sh"
    sur_in = REPO_ROOT / data_cfg.get("surrogate", "data/phase1/surrogate_mask_v1_full")
    merge_into = REPO_ROOT / al_cfg.get(
        "merge_sim_into",
        data_cfg.get("sim_corpus_v1", "data/phase0/sim_results_phase0_v1_all.csv"),
    )
    base_manifest = REPO_ROOT / data_cfg.get("manifest", "data/phase0/manifest.csv")

    if not args.skip_presearch:
        run(
            [
                str(py),
                "scripts/surrogate_ranked_presearch.py",
                "--config",
                str(args.config),
                "--surrogate",
                str(sur_in.relative_to(REPO_ROOT)),
                "--output-dir",
                str(pre_dir.relative_to(REPO_ROOT)),
                "--n-proposals",
                str(pre_cfg.get("n_proposals", 800)),
                "--top-k",
                str(pre_cfg.get("top_k", 40)),
            ],
            dry_run=args.dry_run,
        )

    cand = pre_dir / "top_candidates.csv"
    meep_k = int(pre_cfg.get("meep_verify_top_k", 12))
    meep_manifest = round_dir / "meep_verify.csv"

    if not args.skip_meep:
        if not args.dry_run:
            top = pd.read_csv(cand).head(meep_k)
            top.to_csv(meep_manifest, index=False)
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
                str((round_dir / "meep_verify_results.csv").relative_to(REPO_ROOT)),
                "--no-skip-existing",
                "--force-resim",
            ],
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            new_sim = pd.read_csv(round_dir / "meep_verify_results.csv")
            append_sim_results(merge_into, new_sim, out_path=merge_into)

    manifest_train = round_dir / "manifest_train.csv"
    train_src = meep_manifest if meep_manifest.exists() else cand

    sur_out = round_dir / "surrogate"
    train_ok = args.skip_train and (sur_out / "surrogate.joblib").exists()
    ranking_ok = (round_dir / "ranking_eval.json").exists()

    if not args.skip_train and not args.dry_run:
        build_al_training_manifest(
            REPO_ROOT,
            base_manifest,
            train_src,
            manifest_train,
            base_source_filter="all",
        )
        try:
            run(
                [
                    str(py),
                    "scripts/train_surrogate.py",
                    "--sim-results",
                    str(merge_into.relative_to(REPO_ROOT)),
                    "--manifest",
                    str(manifest_train.relative_to(REPO_ROOT)),
                    "--architecture",
                    al_cfg.get("architecture", "mask_mlp"),
                    "--sources",
                    al_cfg.get("source_filter", "all"),
                    "--output-dir",
                    str(sur_out.relative_to(REPO_ROOT)),
                    "--min-ok",
                    "50",
                ],
                dry_run=False,
            )
            train_ok = True
        except subprocess.CalledProcessError as exc:
            print(f"train_surrogate failed: {exc}", file=sys.stderr)

        if train_ok and (sur_out / "surrogate.joblib").exists() and not args.skip_ranking:
            try:
                run(
                    [
                        str(py),
                        "scripts/evaluate_surrogate_ranking.py",
                        "--surrogate",
                        str(sur_out.relative_to(REPO_ROOT)),
                        "--sim-results",
                        str(merge_into.relative_to(REPO_ROOT)),
                        "--sources",
                        "all",
                        "--output",
                        str((round_dir / "ranking_eval.json").relative_to(REPO_ROOT)),
                    ],
                    dry_run=False,
                )
                ranking_ok = True
            except subprocess.CalledProcessError as exc:
                print(f"ranking eval failed: {exc}", file=sys.stderr)

    summary = {
        "round": args.round,
        "round_dir": str(round_dir.relative_to(REPO_ROOT)),
        "merge_sim_into": str(merge_into.relative_to(REPO_ROOT)),
        "train_ok": train_ok,
        "ranking_ok": ranking_ok,
        "surrogate_out": str(sur_out.relative_to(REPO_ROOT)),
    }
    if (round_dir / "ranking_eval.json").exists() and not args.dry_run:
        summary["ranking_eval"] = json.loads((round_dir / "ranking_eval.json").read_text())
    if not args.dry_run:
        (round_dir / "round_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
