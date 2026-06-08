#!/usr/bin/env python3
"""Promotion panel: prod r25 + sdf_geom dual-pass + DRC for candidate vs champion.

  bash scripts/run_meep.sh scripts/verify_promotion_candidate.py \\
    --candidate-mask data/phase1/wedge_a/meep_gated_shortlist/masks/cand_000160_mask.npy \\
    --candidate-id cand_000160 \\
    --compare-id local_00022
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nano_inv.drc_heuristic import check_mask_heuristic
from nano_inv.meep_sim import MeepRecipe, simulate_mask

SDF_SMOOTH = 0.04
DEFAULT_GATE = REPO / "configs/promote_sdf_geom.yaml"
OUT_DIR = REPO / "data/phase1/wedge_a/meep_gated_shortlist"

CHAMPION_MASKS = {
    "local_00022": REPO / "data/phase1/meep_search_local/candidates/masks/local_00022_mask.npy",
    "meep_bo_00128": REPO / "data/phase1/meep_search_deep/candidates/masks/meep_bo_00128_mask.npy",
    "meep_bo_00093": REPO / "data/phase0/meep_search_100/candidates/masks/meep_bo_00093_mask.npy",
}


def run_panel(mask: np.ndarray, sample_key: str) -> dict:
    drc = check_mask_heuristic(mask)
    rows: list[dict] = []
    prod_recipe = MeepRecipe.for_version("phase0_v1", {"resolution": 25})
    print(f"==> {sample_key} prod_r25")
    prod = simulate_mask(mask, prod_recipe, sample_key=f"{sample_key}_prod_r25")
    rows.append(
        {
            "experiment": "prod_r25",
            "status": prod.status,
            "split_ratio_upper": float(prod.split_ratio_upper) if prod.status == "ok" else None,
        }
    )
    for res in (25, 50):
        label = f"sdf_geom_r{res}"
        recipe = MeepRecipe.for_version(
            "phase0_v1_sdf_geom", {"resolution": res, "sdf_smooth_um": SDF_SMOOTH}
        )
        print(f"    {label}")
        r = simulate_mask(mask, recipe, sample_key=f"{sample_key}_{label}")
        rows.append(
            {
                "experiment": label,
                "status": r.status,
                "split_ratio_upper": float(r.split_ratio_upper) if r.status == "ok" else None,
            }
        )
    prod_split = rows[0]["split_ratio_upper"]
    geom25 = next(x["split_ratio_upper"] for x in rows if x["experiment"] == "sdf_geom_r25")
    geom50 = next(x["split_ratio_upper"] for x in rows if x["experiment"] == "sdf_geom_r50")
    gap = abs(geom50 - geom25) if geom25 is not None and geom50 is not None else float("nan")
    dprod = abs(geom25 - prod_split) if prod_split is not None and geom25 is not None else float("nan")
    return {
        "sample_id": sample_key,
        "drc_pass": bool(drc.passed),
        "drc_reasons": drc.reasons,
        "meep": rows,
        "mesh_gap": gap,
        "prod_geom_delta": dprod,
    }


def gate_pass(entry: dict, *, max_gap: float, max_dprod: float) -> bool:
    if not entry["drc_pass"]:
        return False
    if entry["mesh_gap"] != entry["mesh_gap"] or entry["mesh_gap"] > max_gap:
        return False
    if entry["prod_geom_delta"] != entry["prod_geom_delta"] or entry["prod_geom_delta"] > max_dprod:
        return False
    prod = entry["meep"][0]["split_ratio_upper"]
    if prod is None or abs(prod - 0.5) > 0.05:
        return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-mask", type=Path, required=True)
    p.add_argument("--candidate-id", default="candidate")
    p.add_argument("--compare-id", default="local_00022")
    p.add_argument("--gate-config", type=Path, default=DEFAULT_GATE)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    gate_cfg = yaml.safe_load(args.gate_config.read_text())["gate"]
    max_gap = float(gate_cfg["max_mesh_gap"])
    max_dprod = float(gate_cfg["max_prod_delta"])

    cand_mask = np.load(args.candidate_mask if args.candidate_mask.is_absolute() else REPO / args.candidate_mask)
    results: list[dict] = []
    results.append(run_panel(cand_mask, args.candidate_id))

    compare_path = CHAMPION_MASKS.get(args.compare_id)
    if compare_path and compare_path.exists():
        results.append(run_panel(np.load(compare_path), args.compare_id))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"promotion_panel_{args.candidate_id}.json"
    out_json.write_text(json.dumps(results, indent=2))

    lines = [
        f"# Promotion panel: `{args.candidate_id}` vs `{args.compare_id}`",
        "",
        f"Gate (sdf_geom): mesh gap ≤ {max_gap}, |sdf_geom r25 − prod| ≤ {max_dprod}, DRC pass",
        "",
        "| sample | DRC | prod r25 | sdf_geom r25 | sdf_geom r50 | gap | |geom−prod| | promotable? |",
        "|--------|-----|----------|--------------|--------------|-----|------------|-------------|",
    ]
    for e in results:
        prod = e["meep"][0]["split_ratio_upper"]
        g25 = next(x["split_ratio_upper"] for x in e["meep"] if x["experiment"] == "sdf_geom_r25")
        g50 = next(x["split_ratio_upper"] for x in e["meep"] if x["experiment"] == "sdf_geom_r50")
        ok = gate_pass(e, max_gap=max_gap, max_dprod=max_dprod)
        lines.append(
            f"| {e['sample_id']} | {'✓' if e['drc_pass'] else '—'} | "
            f"{prod:.4f} | {g25:.4f} | {g50:.4f} | {e['mesh_gap']:.3f} | {e['prod_geom_delta']:.3f} | "
            f"{'✓' if ok else '—'} |"
        )
    out_md = args.out_dir / f"promotion_panel_{args.candidate_id}.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_json}\nWrote {out_md}")


if __name__ == "__main__":
    main()
