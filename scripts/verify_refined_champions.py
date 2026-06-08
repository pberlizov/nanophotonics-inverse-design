#!/usr/bin/env python3
"""MEEP + gym hard-mask check: baseline champion z vs refined z.

  # Gym-refined (Path A exploratory):
  PYTHONPATH=src python scripts/verify_refined_champions.py --gym-only
  bash scripts/run_meep.sh scripts/verify_refined_champions.py --meep-only --refine-source gym

  # Surrogate-refined (product path):
  PYTHONPATH=src python scripts/verify_refined_champions.py --refine-source surrogate --gym-only
  bash scripts/run_meep.sh scripts/verify_refined_champions.py --meep-only --refine-source surrogate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nano_inv.latent import pad_latent_to_standard
from nano_inv.meep_sim import MeepRecipe, simulate_mask

REFINE_SOURCES = {
    "gym": {
        "refine_json": REPO / "data/phase1/invrs_benchmark/refine_champion_grad.json",
        "out_dir": REPO / "data/phase1/invrs_benchmark",
        "mask_subdir": "refined_masks",
        "refine_metric_key": "final_eval_metric",
        "title": "gym soft refine (Path A)",
    },
    "surrogate": {
        "refine_json": REPO / "data/phase1/refine_surrogate/refine_champion_surrogate.json",
        "out_dir": REPO / "data/phase1/refine_surrogate",
        "mask_subdir": "verify_masks",
        "refine_metric_key": "final_pred_split",
        "title": "surrogate refine (MEEP-aligned)",
    },
}


def resolve_paths(args: argparse.Namespace) -> dict:
    src = args.refine_source
    base = REFINE_SOURCES[src].copy()
    if args.refine_json:
        base["refine_json"] = args.refine_json if args.refine_json.is_absolute() else REPO / args.refine_json
    if args.refine_dir:
        base["out_dir"] = args.refine_dir if args.refine_dir.is_absolute() else REPO / args.refine_dir
    base["mask_dir"] = base["out_dir"] / base["mask_subdir"]
    suffix = args.latent_suffix or ("champions" if src == "gym" else src)
    base["out_json"] = base["out_dir"] / f"verify_refined_{suffix}.json"
    base["out_md"] = base["out_dir"] / f"verify_refined_{suffix}.md"
    base["gym_cache"] = base["out_dir"] / f"verify_gym_cache_{suffix}.json"
    base["partial_json"] = base["out_dir"] / f"verify_refined_partial_{suffix}.json"
    return base


SDF_SMOOTH = 0.04
MEEP_EXPERIMENTS = [
    ("prod_r25", "phase0_v1", 25, {}),
    ("sdf_geom_r25", "phase0_v1_sdf_geom", 25, {"sdf_smooth_um": SDF_SMOOTH}),
    ("sdf_geom_r50", "phase0_v1_sdf_geom", 50, {"sdf_smooth_um": SDF_SMOOTH}),
]


def load_refine_rows(refine_json: Path) -> list[dict]:
    if not refine_json.exists():
        raise FileNotFoundError(f"Missing refine report: {refine_json}")
    return json.loads(refine_json.read_text())


def latent_to_mask(latent: np.ndarray, manifold) -> np.ndarray:
    z = pad_latent_to_standard(latent)
    if z.ndim == 3:
        z = z[None, ...]
    mask = manifold.decode_numpy(z)
    return (mask > 0.5).astype(np.float32)


def mask_fraction_changed(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    return float(np.mean(a != b))


def run_gym_hard(mask: np.ndarray, challenge: str) -> dict:
    from nano_inv.invrs_adapter import evaluate_gym_params, mask_to_gym_params

    params = mask_to_gym_params(mask, challenge=challenge, use_soft=False)  # type: ignore[arg-type]
    return evaluate_gym_params(params, challenge=challenge)  # type: ignore[arg-type]


def load_gym_cache(gym_cache: Path) -> dict[str, dict]:
    if not gym_cache.exists():
        return {}
    data = json.loads(gym_cache.read_text())
    return {row["sample_id"]: row for row in data}


def save_gym_cache(report: list[dict], gym_cache: Path) -> None:
    cache = []
    for row in report:
        cache.append(
            {
                "sample_id": row["sample_id"],
                "mask_frac_changed": row.get("mask_frac_changed"),
                "refine_metric": row.get("refine_metric"),
                "gym_hard_baseline": row["variants"]["baseline"].get("gym_hard"),
                "gym_hard_refined": row["variants"]["refined"].get("gym_hard"),
            }
        )
    gym_cache.write_text(json.dumps(cache, indent=2))


def merge_gym_cache(report: list[dict], gym_cache: Path) -> None:
    cache = load_gym_cache(gym_cache)
    for row in report:
        sid = row["sample_id"]
        if sid not in cache:
            continue
        c = cache[sid]
        row.setdefault("mask_frac_changed", c.get("mask_frac_changed"))
        row.setdefault("refine_metric", c.get("refine_metric") or c.get("gym_refine_eval_metric"))
        for variant, key in (("baseline", "gym_hard_baseline"), ("refined", "gym_hard_refined")):
            if c.get(key):
                row["variants"].setdefault(variant, {})["gym_hard"] = c[key]


def compute_meep_delta(entry: dict) -> None:
    base_meep = {
        m["experiment"]: m["split_ratio_upper"]
        for m in entry["variants"]["baseline"].get("meep", [])
        if m["status"] == "ok"
    }
    ref_meep = {
        m["experiment"]: m["split_ratio_upper"]
        for m in entry["variants"]["refined"].get("meep", [])
        if m["status"] == "ok"
    }
    entry["meep_delta"] = {
        k: (ref_meep.get(k), base_meep.get(k), (ref_meep.get(k) or 0) - (base_meep.get(k) or 0))
        for k in sorted(set(base_meep) | set(ref_meep))
    }


def _fmt_opt(v: float | None, fmt: str = ".3f") -> str:
    return format(v, fmt) if v is not None else "—"


def write_report(
    report: list[dict],
    *,
    challenge: str,
    paths: dict,
    refine_source: str,
) -> None:
    merge_gym_cache(report, paths["gym_cache"])
    for entry in report:
        if entry["variants"]["baseline"].get("meep") and entry["variants"]["refined"].get("meep"):
            compute_meep_delta(entry)

    paths["out_json"].write_text(json.dumps(report, indent=2))

    refine_label = paths.get("refine_metric_key", "refine_metric")
    refine_col = "surrogate pred" if refine_source == "surrogate" else "gym soft"
    lines = [
        f"# Refined champion verification ({paths['title']})",
        "",
        f"Challenge gym: `{challenge}` · MEEP: prod r25 + sdf_geom r25/r50 (smooth={SDF_SMOOTH})",
        "",
        f"| sample | mask Δ% | {refine_col} | gym hard base → ref | prod r25 base → ref | sdf_geom r25 | gap ref |",
        "|--------|---------|----------|----------------------|---------------------|--------------|---------|",
    ]
    for e in report:
        gb = e["variants"]["baseline"].get("gym_hard", {}).get("eval_metric")
        gr = e["variants"]["refined"].get("gym_hard", {}).get("eval_metric")
        rm = e.get("refine_metric")
        gym_col = (
            f"{gb:.1f} → {gr:.1f}"
            if gb is not None and gr is not None
            else "—"
        )
        meep_d = e.get("meep_delta") or {}
        prod = meep_d.get("prod_r25", (None, None, None))
        geom25 = meep_d.get("sdf_geom_r25", (None, None, None))
        geom50_ref = next(
            (
                m["split_ratio_upper"]
                for m in e["variants"]["refined"].get("meep", [])
                if m["experiment"] == "sdf_geom_r50" and m["status"] == "ok"
            ),
            None,
        )
        geom25_ref_v = next(
            (
                m["split_ratio_upper"]
                for m in e["variants"]["refined"].get("meep", [])
                if m["experiment"] == "sdf_geom_r25" and m["status"] == "ok"
            ),
            None,
        )
        gap = (
            abs(geom50_ref - geom25_ref_v)
            if geom50_ref is not None and geom25_ref_v is not None
            else float("nan")
        )
        prod_col = (
            f"{prod[1]:.3f} → {prod[0]:.3f} (Δ{prod[2]:+.3f})"
            if prod[1] is not None and prod[0] is not None
            else "—"
        )
        geom_col = (
            f"{geom25[1]:.3f} → {geom25[0]:.3f}"
            if geom25[1] is not None and geom25[0] is not None
            else "—"
        )
        lines.append(
            f"| {e['sample_id']} | {e['mask_frac_changed']*100:.1f}% | "
            f"{_fmt_opt(rm)} | {gym_col} | {prod_col} | {geom_col} | "
            f"{_fmt_opt(gap if gap == gap else None)} |"
        )

    lines.extend(["", "## Interpretation", ""])
    if refine_source == "surrogate":
        lines.extend(
            [
                "- **Surrogate refine** minimizes `(pred_split − 0.5)²` on the MEEP-trained **mask_mlp** ranker (STE/hard decode; trust region on z).",
                "- **`phase0_v1` prod r25** is the product gate (|split−0.5| < 0.03, mask Δ < 10%); **`sdf_geom`** is mesh-stability only.",
                "- Compare prod r25 Δ vs baseline; do not promote on surrogate pred or sdf_geom alone.",
            ]
        )
    else:
        lines.extend(
            [
                "- **Gym soft refine** (Adam on z) reached in_spec (~0.01–0.02); **hard gym** decode stays poor (~−4 to −9 vs ~−34 baseline).",
                "- **`sdf_geom` MEEP** stays **0.500 / 0.500** (gap 0) for baseline and refined — mesh-stable recipe does not reflect prod-r25 failure.",
                "- **`phase0_v1` prod r25** is the product gate: only **`local_00022`** stays near 50/50 (Δ≈+0.02); **`meep_bo_*`** collapse to ~**0.97–0.99** upper arm.",
                "- **Do not promote** gym-refined latents; use **surrogate refine + prod MEEP verify** for the product path.",
            ]
        )
    paths["out_md"].write_text("\n".join(lines) + "\n")


def run_meep_panel(mask: np.ndarray, sample_key: str) -> list[dict]:
    rows: list[dict] = []
    for label, version, res, extra in MEEP_EXPERIMENTS:
        recipe = MeepRecipe.for_version(version, {"resolution": res, **extra})
        print(f"    MEEP {label} …")
        r = simulate_mask(mask, recipe, sample_key=f"{sample_key}_{label}")
        rows.append(
            {
                "experiment": label,
                "recipe_version": version,
                "resolution": res,
                "status": r.status,
                "split_ratio_upper": float(r.split_ratio_upper) if r.status == "ok" else None,
                "error": r.error,
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-id", default=None)
    p.add_argument(
        "--refine-source",
        choices=("gym", "surrogate"),
        default="gym",
        help="Which refine report / output dir to verify",
    )
    p.add_argument(
        "--refine-json",
        type=Path,
        default=None,
        help="Override refine JSON (default from --refine-source)",
    )
    p.add_argument(
        "--refine-dir",
        type=Path,
        default=None,
        help="Override verify output dir (default from --refine-source)",
    )
    p.add_argument(
        "--latent-suffix",
        default=None,
        help="Suffix for verify_refined_{suffix}.json/md (default: refine-source name)",
    )
    p.add_argument("--gym-only", action="store_true", help="Skip MEEP (fast sanity)")
    p.add_argument(
        "--meep-only",
        action="store_true",
        help="MEEP from saved verify_masks/ (no JAX; use after --gym-only)",
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Merge gym cache + existing verify JSON into final report",
    )
    p.add_argument("--challenge", default="ceviche_lightweight_power_splitter")
    args = p.parse_args()

    paths = resolve_paths(args)
    paths["refine_metric_key"] = REFINE_SOURCES[args.refine_source]["refine_metric_key"]

    if args.report_only:
        if not paths["out_json"].exists():
            raise FileNotFoundError(f"Missing {paths['out_json']}")
        report = json.loads(paths["out_json"].read_text())
        write_report(report, challenge=args.challenge, paths=paths, refine_source=args.refine_source)
        print(f"Wrote {paths['out_json']}\nWrote {paths['out_md']}")
        return

    challenge = args.challenge
    rows_cfg = load_refine_rows(paths["refine_json"])
    if args.sample_id:
        rows_cfg = [r for r in rows_cfg if r["sample_id"] == args.sample_id]

    mask_dir = paths["mask_dir"]
    mask_dir.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict] = {}
    if paths["out_json"].exists() and (args.gym_only or args.meep_only):
        for row in json.loads(paths["out_json"].read_text()):
            existing[row["sample_id"]] = row

    metric_key = paths["refine_metric_key"]
    report: list[dict] = []
    manifold = None
    if not args.meep_only:
        from nano_inv.manifold import EBeamManifold

        manifold = EBeamManifold.load()

    for row in rows_cfg:
        sid = row["sample_id"]
        print(f"==> {sid}")
        base_path = mask_dir / f"{sid}_baseline_mask.npy"
        ref_path = mask_dir / f"{sid}_refined_mask.npy"
        if args.meep_only:
            if not base_path.exists() or not ref_path.exists():
                print(f"    SKIP: run without --meep-only first (missing {mask_dir})")
                continue
            mask_base = np.load(base_path)
            mask_ref = np.load(ref_path)
            frac = mask_fraction_changed(mask_base, mask_ref)
        else:
            z_base = np.load(REPO / row["latent_path"])
            z_ref = np.load(REPO / row["refined_latent_path"])
            mask_base = latent_to_mask(z_base, manifold)
            mask_ref = latent_to_mask(z_ref, manifold)
            frac = mask_fraction_changed(mask_base, mask_ref)
            np.save(base_path, mask_base)
            np.save(ref_path, mask_ref)
        print(f"    mask pixels changed: {frac * 100:.2f}%")

        entry: dict = {
            "sample_id": sid,
            "refine_source": args.refine_source,
            "mask_frac_changed": frac,
            "refine_metric": row.get(metric_key),
            "variants": {},
        }

        for variant, mask in (("baseline", mask_base), ("refined", mask_ref)):
            variant_row: dict = {}
            if not args.meep_only:
                print(f"    gym hard ({variant}) …")
                gym = run_gym_hard(mask, challenge)
                variant_row["gym_hard"] = gym
                print(
                    f"      gym eval_metric={gym['eval_metric']:.4f}  "
                    f"in_spec={gym['in_spec']}"
                )
            elif sid in existing:
                prior = existing[sid]["variants"].get(variant, {})
                if prior.get("gym_hard"):
                    variant_row["gym_hard"] = prior["gym_hard"]
            if not args.gym_only:
                variant_row["meep"] = run_meep_panel(mask, f"{sid}_{variant}")
            elif sid in existing:
                prior = existing[sid]["variants"].get(variant, {})
                if prior.get("meep"):
                    variant_row["meep"] = prior["meep"]
            entry["variants"][variant] = variant_row

        report.append(entry)
        paths["partial_json"].write_text(json.dumps(report, indent=2))

    if args.gym_only:
        save_gym_cache(report, paths["gym_cache"])
    write_report(report, challenge=challenge, paths=paths, refine_source=args.refine_source)
    print(f"\nWrote {paths['out_json']}\nWrote {paths['out_md']}")


if __name__ == "__main__":
    main()
