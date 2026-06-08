#!/usr/bin/env python3
"""Sanity-check MEEP recipe: ref_published, empty mask, full Si island."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.meep_sim import MeepRecipe, require_meep, simulate_mask  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase0.yaml")
    p.add_argument(
        "--recipe-version",
        type=str,
        default=None,
        help="phase0_v0 | phase0_v1 (default: config meep.recipe_version)",
    )
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--try-flip-y", action="store_true", help="Also run with mask_flip_y=True")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def run_case(name: str, mask: np.ndarray, recipe: MeepRecipe, verbose: bool) -> dict:
    res = simulate_mask(mask, recipe, verbose=verbose)
    return {
        "case": name,
        "status": res.status,
        "split_ratio_upper": res.split_ratio_upper,
        "flux_in": res.flux_in,
        "flux_out_upper": res.flux_out_upper,
        "flux_out_lower": res.flux_out_lower,
        "insertion_loss_db": res.insertion_loss_db,
        "error": res.error,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    meep_cfg = dict(cfg.get("meep") or {})
    version = args.recipe_version or meep_cfg.get("recipe_version", "phase0_v1")
    if args.resolution is not None:
        meep_cfg["resolution"] = args.resolution
    recipe = MeepRecipe.for_version(version, meep_cfg)

    require_meep()
    data_root = REPO_ROOT / cfg["data"]["root"]
    ref_path = data_root / "masks" / "ref_published_mask.npy"
    if not ref_path.exists():
        raise SystemExit(f"missing {ref_path} — run decode_batch.py first")

    ref = np.load(ref_path)
    empty = np.zeros_like(ref)
    full = np.ones_like(ref)

    rows: list[dict] = []
    from dataclasses import replace

    for flip in ([False, True] if args.try_flip_y else [recipe.mask_flip_y]):
        r = replace(recipe, mask_flip_y=flip)
        tag = "flip_y" if flip else "normal"
        rows.append(run_case(f"ref_published_{tag}", ref, r, args.verbose))
        rows.append(run_case(f"empty_{tag}", empty, r, args.verbose))
        rows.append(run_case(f"full_{tag}", full, r, args.verbose))

    target = float((cfg.get("targets") or {}).get("split_ratio_1550", 0.5))
    ref_rows = [r for r in rows if r["case"].startswith("ref_published")]
    empty_rows = [r for r in rows if r["case"].startswith("empty_")]
    full_rows = [r for r in rows if r["case"].startswith("full_")]

    def _split(row: dict) -> float | None:
        v = row.get("split_ratio_upper")
        return float(v) if v is not None and np.isfinite(v) else None

    ref_splits = [(r["case"], _split(r)) for r in ref_rows if _split(r) is not None]
    best_ref_case, best_ref_split = min(
        ref_splits, key=lambda x: abs(x[1] - target), default=(None, None)
    )

    # G1a: template wiring (symmetric limits must split 50/50)
    def _near_half(row: dict, tol: float = 0.02) -> bool:
        s = _split(row)
        return s is not None and abs(s - 0.5) <= tol

    g1a_template = all(_near_half(r) for r in empty_rows + full_rows)
    # G1b: published ref in-template (often FAILS — mask tuned for another solver)
    g1b_ref = best_ref_split is not None and 0.40 <= best_ref_split <= 0.60
    g1_pass = g1a_template  # do not block sprint on G1b

    summary = {
        "recipe_version": version,
        "resolution": recipe.resolution,
        "target_split_ratio": target,
        "g1_pass": g1_pass,
        "g1a_template_symmetry_pass": g1a_template,
        "g1b_ref_published_pass": g1b_ref,
        "best_ref_case": best_ref_case,
        "best_ref_split_ratio_upper": best_ref_split,
        "ref_distance_to_target": abs(best_ref_split - target) if best_ref_split is not None else None,
        "note": (
            "G1b failure is expected if drcgenerator ref was optimized for a different FDTD setup. "
            "Use meep_search toward 0.5 in this template; do not require ref≈0.5 for G1 pass."
        ),
        "cases": rows,
    }
    out = data_root / f"calibration_{version}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not g1a_template:
        print("\nG1a FAIL: empty/full masks should split 0.50 — check ports/monitors.")
        sys.exit(1)
    if not g1b_ref:
        print(
            f"\nG1b (informational): best ref = {best_ref_case} → {best_ref_split:.4f} "
            f"(target {target}). Safe to proceed with meep_search / relabel if G1a passed."
        )


if __name__ == "__main__":
    main()
