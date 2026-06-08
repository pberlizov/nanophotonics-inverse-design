#!/usr/bin/env python3
"""Label Phase 0 masks with MEEP (2D TE splitter template). Requires conda env `mp`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.meep_sim import MeepRecipe, MeepSimResult, require_meep, simulate_mask  # noqa: E402
from nano_inv.manifest import filter_by_source  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase0.yaml")
    p.add_argument("--manifest", type=Path, default=None, help="Override manifest path")
    p.add_argument("--limit", type=int, default=None, help="Max sims this run")
    p.add_argument("--only-pass-drc", action="store_true", default=True)
    p.add_argument("--no-only-pass-drc", action="store_false", dest="only_pass_drc")
    p.add_argument("--resolution", type=int, default=None, help="Override meep.resolution")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="List jobs only")
    p.add_argument(
        "--sample-ids",
        type=str,
        default=None,
        help="Comma-separated sample_ids to run (overrides --limit)",
    )
    p.add_argument(
        "--force-resim",
        action="store_true",
        help="Re-run even if sample_id already has status=ok",
    )
    p.add_argument(
        "--recipe-version",
        type=str,
        default=None,
        help="phase0_v0 | phase0_v1 (default: config meep.recipe_version)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: data/phase0/sim_results.csv)",
    )
    p.add_argument(
        "--sources-filter",
        type=str,
        default="all",
        help="Manifest source filter: all | perturb | perlin",
    )
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = REPO_ROOT / cfg["data"]["root"]
    manifest_path = Path(args.manifest) if args.manifest else REPO_ROOT / cfg["data"]["manifest"]
    out_path = Path(args.output) if args.output else data_root / "sim_results.csv"
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    meep_cfg = dict(cfg.get("meep") or {})
    if args.resolution is not None:
        meep_cfg["resolution"] = args.resolution
    recipe_version = args.recipe_version or meep_cfg.get("recipe_version", "phase0_v0")
    recipe = MeepRecipe.for_version(recipe_version, meep_cfg)

    df = pd.read_csv(manifest_path)
    if args.only_pass_drc and "drc_heuristic_pass" in df.columns:
        df = df[df["drc_heuristic_pass"] == True]  # noqa: E712
    df = filter_by_source(df, args.sources_filter)

    if args.sample_ids:
        wanted = {s.strip() for s in args.sample_ids.split(",") if s.strip()}
        df = df[df["sample_id"].astype(str).isin(wanted)]
    elif args.limit is not None:
        df = df.head(args.limit)

    done_ids: set[str] = set()
    if args.skip_existing and not args.force_resim and out_path.exists():
        prev = pd.read_csv(out_path)
        done_ids = set(prev.loc[prev["status"] == "ok", "sample_id"].astype(str))

    require_meep()

    jobs = []
    for _, row in df.iterrows():
        sid = str(row["sample_id"])
        if sid in done_ids:
            continue
        jobs.append(row)

    if args.dry_run:
        print(f"recipe_version={recipe_version} resolution={recipe.resolution}")
        print(f"would run {len(jobs)} sims (skip {len(done_ids)} existing ok)")
        for row in jobs[:10]:
            print(" ", row["sample_id"], row["mask_path"])
        return

    if not jobs:
        raise SystemExit(
            "no MEEP jobs to run — check --manifest / --sample-ids / --sources-filter"
        )

    rows_out: list[dict] = []
    if out_path.exists():
        prev_df = pd.read_csv(out_path)
        if args.force_resim:
            resim_ids = set(df["sample_id"].astype(str))
            prev_df = prev_df[~prev_df["sample_id"].astype(str).isin(resim_ids)]
        rows_out = prev_df.to_dict("records")

    for row in tqdm(jobs, desc="meep"):
        sid = str(row["sample_id"])
        mask_path = REPO_ROOT / row["mask_path"]
        mask = np.load(mask_path)

        try:
            res = simulate_mask(mask, recipe, verbose=args.verbose)
        except Exception as e:
            import traceback

            err = str(e).strip() or repr(e)
            if args.verbose:
                traceback.print_exc()
            res = MeepSimResult(
                status="error",
                flux_in=float("nan"),
                flux_out_upper=float("nan"),
                flux_out_lower=float("nan"),
                split_ratio_upper=float("nan"),
                insertion_loss_db=float("nan"),
                error=err[:2000],
            )

        rows_out.append(
            {
                "sample_id": sid,
                "source": row.get("source", ""),
                "mask_path": row["mask_path"],
                "recipe_version": recipe_version,
                "resolution": recipe.resolution,
                "status": res.status,
                "flux_in": res.flux_in,
                "flux_out_upper": res.flux_out_upper,
                "flux_out_lower": res.flux_out_lower,
                "split_ratio_upper": res.split_ratio_upper,
                "insertion_loss_db": res.insertion_loss_db,
                "error": getattr(res, "error", ""),
            }
        )

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_path, index=False)

    ok = out_df[out_df["status"] == "ok"]
    summary = {
        "recipe_version": recipe_version,
        "n_total": len(out_df),
        "n_ok": len(ok),
        "split_ratio_mean": float(ok["split_ratio_upper"].mean()) if len(ok) else None,
        "split_ratio_std": float(ok["split_ratio_upper"].std()) if len(ok) else None,
        "output": str(out_path.relative_to(REPO_ROOT)),
    }
    (data_root / "sim_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
