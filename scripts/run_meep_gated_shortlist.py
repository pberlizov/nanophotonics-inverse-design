#!/usr/bin/env python3
"""MEEP prod r25 on top-K surrogate-ranked proposals (search loop spike).

  PYTHONPATH=src python scripts/run_meep_gated_shortlist.py \\
    --candidates data/phase1/refine_surrogate/smoke_ranked_50.csv --top-k 5

  # Or generate + verify in one shot:
  PYTHONPATH=src python scripts/run_meep_gated_shortlist.py --n-proposals 200 --top-k 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv/bin/python"
DEFAULT_OUT = REPO / "data/phase1/wedge_a/meep_gated_shortlist"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO / "configs/wedge_a_improved.yaml")
    p.add_argument("--candidates", type=Path, default=None, help="Ranked CSV (sample_id, mask_path, surrogate_score)")
    p.add_argument("--n-proposals", type=int, default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--surrogate", type=Path, default=REPO / "data/phase1/wedge_a/surrogate_improved")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42, help="Passed to generate_ranked_candidates")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = REPO.resolve()
    cand_path = args.candidates

    if cand_path is None:
        if args.n_proposals is None:
            raise SystemExit("Provide --candidates or --n-proposals")
        cand_path = out_dir / f"ranked_{args.n_proposals}.csv"
        out_rel = cand_path.resolve().relative_to(repo)
        gen_cmd = [
            str(PY) if PY.exists() else sys.executable,
            str(REPO / "scripts/generate_ranked_candidates.py"),
            "--config",
            str(args.config),
            "--surrogate",
            str(args.surrogate),
            "--n-proposals",
            str(args.n_proposals),
            "--output",
            str(out_rel),
            "--seed",
            str(args.seed),
        ]
        print("+", " ".join(gen_cmd))
        subprocess.run(gen_cmd, cwd=REPO, check=True, env={**__import__("os").environ, "PYTHONPATH": "src"})

    df = pd.read_csv(REPO / cand_path if not cand_path.is_absolute() else cand_path)
    if "surrogate_score" in df.columns:
        df = df.sort_values("surrogate_score", ascending=True)
    top = df.head(args.top_k).copy()
    manifest = out_dir / f"meep_manifest_top{args.top_k}.csv"
    top.to_csv(manifest, index=False)
    meep_out = out_dir / f"meep_prod_r25_top{args.top_k}.csv"

    if args.dry_run:
        print(f"Would MEEP verify {len(top)} masks → {meep_out}")
        print(top[["sample_id", "pred_split_ratio_upper", "surrogate_score"]].to_string())
        return

    meep_cmd = [
        "bash",
        str(REPO / "scripts/run_meep.sh"),
        str(REPO / "scripts/run_fdtd_batch.py"),
        "--config",
        str(REPO / "configs/phase0.yaml"),
        "--manifest",
        str(manifest.resolve().relative_to(repo)),
        "--output",
        str(meep_out.resolve().relative_to(repo)),
        "--recipe-version",
        "phase0_v1",
        "--resolution",
        "25",
        "--no-skip-existing",
        "--force-resim",
    ]
    print("+", " ".join(meep_cmd))
    subprocess.run(meep_cmd, cwd=REPO, check=True)

    res = pd.read_csv(meep_out)
    ok = res[res["status"] == "ok"].copy()
    ok["abs_err"] = (ok["split_ratio_upper"] - 0.5).abs()
    best = ok.sort_values("abs_err").head(10)
    summary = {
        "top_k": args.top_k,
        "manifest": str(manifest.relative_to(REPO)),
        "meep_results": str(meep_out.relative_to(REPO)),
        "n_ok": int(len(ok)),
        "best": best[["sample_id", "split_ratio_upper", "abs_err"]].to_dict("records"),
    }
    (out_dir / "meep_gated_summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        f"# MEEP-gated shortlist (prod r25, top {args.top_k})",
        "",
        f"Candidates: `{cand_path.relative_to(REPO) if cand_path.is_relative_to(REPO) else cand_path}`",
        "",
        "| sample_id | prod r25 | |err−0.5| |",
        "|-----------|----------|---------|",
    ]
    for _, r in ok.sort_values("abs_err").iterrows():
        lines.append(
            f"| {r['sample_id']} | {r['split_ratio_upper']:.4f} | {r['abs_err']:.4f} |"
        )
    (out_dir / "meep_gated_summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_dir / 'meep_gated_summary.md'}")


if __name__ == "__main__":
    main()
