#!/usr/bin/env python3
"""Re-score existing local-search candidates on a C-band λ grid (no Optuna)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nano_inv.meep_sim import MeepRecipe, require_meep, simulate_mask_broadband  # noqa: E402

OUT = REPO / "data/phase1/release"


def repo_relative(path: Path) -> str:
    p = (REPO / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        return str(p.relative_to(REPO.resolve()))
    except ValueError:
        return str(p)


def json_safe(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--candidates",
        type=Path,
        default=REPO / "data/phase1/meep_search_local/top_candidates.csv",
    )
    p.add_argument("--trials", type=Path, default=REPO / "data/phase1/meep_search_local/meep_search_local_trials.csv")
    p.add_argument("--use-trials", action="store_true", help="Score all completed trials, not just top-k")
    p.add_argument("--config", type=Path, default=REPO / "configs/broadband_search_local.yaml")
    p.add_argument("--wl-start", type=float, default=1.53)
    p.add_argument("--wl-stop", type=float, default=1.57)
    p.add_argument("--wl-step", type=float, default=0.005)
    p.add_argument("--max-worst-split-error", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", type=Path, default=OUT / "broadband_rescore.json")
    p.add_argument("--resume", action="store_true", help="Skip sample_ids already in output JSON")
    return p.parse_args()


def write_outputs(args: argparse.Namespace, *, src: Path, wls: list[float], results: list[dict]) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json_safe(
        {
            "source_csv": repo_relative(src),
            "wavelengths_um": wls,
            "max_worst_split_error": args.max_worst_split_error,
            "n_pass": sum(1 for r in results if r.get("pass_broadband_gate")),
            "n_total": len(results),
            "n_ok": sum(1 for r in results if r.get("status") == "ok"),
            "results": results,
        }
    )
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output} ({payload['n_pass']}/{payload['n_total']} pass)")

    md = args.output.with_suffix(".md")
    lines = [
        "# Broadband rescore (existing candidates)\n",
        f"Source: `{payload['source_csv']}` | λ {wls[0]}–{wls[-1]} µm ({len(wls)} pts)\n",
        f"Gate: worst |err| ≤ {args.max_worst_split_error}\n\n",
        "| sample_id | worst |err| | pass? |\n|-----------|-------------|-------|\n",
    ]
    ok = [r for r in results if r.get("status") == "ok" and r.get("worst_split_error") is not None]
    for r in sorted(ok, key=lambda x: x["worst_split_error"]):
        lines.append(
            f"| {r['sample_id']} | {r['worst_split_error']:.4f} | "
            f"{'yes' if r.get('pass_broadband_gate') else 'no'} |\n"
        )
    md.write_text("".join(lines))
    print(f"wrote {md}")


def main() -> None:
    args = parse_args()
    require_meep()
    cfg = yaml.safe_load(args.config.read_text())
    meep_cfg = dict(cfg.get("meep") or {})
    recipe = MeepRecipe.for_version(meep_cfg.get("recipe_version", "phase0_v1"), meep_cfg)
    target = float((cfg.get("targets") or {}).get("split_ratio_1550", 0.5))

    src = args.trials if args.use_trials else args.candidates
    if not src.is_absolute():
        src = (REPO / src).resolve()
    df = pd.read_csv(src)
    if "mask_path" not in df.columns and "sample_id" in df.columns:
        root = REPO / "data/phase1/meep_search_local/candidates/masks"
        df["mask_path"] = df["sample_id"].map(lambda s: str(root / f"{s}_mask.npy"))

    wls = [round(float(w), 3) for w in np.arange(args.wl_start, args.wl_stop + 1e-9, args.wl_step)]
    rows = df.head(args.limit) if args.limit else df

    done: dict[str, dict] = {}
    if args.resume and args.output.exists():
        prev = json.loads(args.output.read_text())
        for r in prev.get("results", []):
            sid = r.get("sample_id")
            if sid:
                done[sid] = r
        print(f"resume: {len(done)} results loaded from {args.output}")

    results: list[dict] = []
    for _, row in rows.iterrows():
        sid = str(row.get("sample_id", "unknown"))
        if sid in done:
            results.append(done[sid])
            continue

        rel_mask = str(row["mask_path"])
        mask_path = REPO / rel_mask if not rel_mask.startswith("/") else Path(rel_mask)
        if not mask_path.exists():
            entry = {"sample_id": sid, "status": "missing_mask", "mask_path": repo_relative(mask_path)}
            results.append(entry)
            done[sid] = entry
            continue

        try:
            mask = np.load(mask_path)
            bb = simulate_mask_broadband(mask, recipe, wls, target_split=target, verbose=False)
            passed = (
                bb.status == "ok"
                and np.isfinite(bb.worst_split_error)
                and bb.worst_split_error <= args.max_worst_split_error
            )
            entry = {
                "sample_id": sid,
                "mask_path": repo_relative(mask_path),
                "status": bb.status,
                "worst_split_error": float(bb.worst_split_error) if np.isfinite(bb.worst_split_error) else None,
                "mean_IL_db": float(bb.mean_insertion_loss_db) if np.isfinite(bb.mean_insertion_loss_db) else None,
                "R_up_by_wl": {str(k): float(v) for k, v in bb.splits_by_wavelength.items()},
                "pass_broadband_gate": passed,
                "error": bb.error or "",
            }
            if bb.status == "ok" and entry["worst_split_error"] is not None:
                gate = "PASS" if passed else "FAIL"
                print(f"{sid}: worst |err|={entry['worst_split_error']:.4f} [{gate}]")
            else:
                print(f"{sid}: status={bb.status} error={bb.error or 'sim_failed'}")
                entry["status"] = entry["status"] if entry["status"] != "ok" else "error"
        except Exception as exc:
            entry = {
                "sample_id": sid,
                "mask_path": repo_relative(mask_path),
                "status": "error",
                "error": str(exc),
            }
            print(f"{sid}: ERROR {exc}")

        results.append(entry)
        done[sid] = entry
        write_outputs(args, src=src, wls=wls, results=results)

    write_outputs(args, src=src, wls=wls, results=results)


if __name__ == "__main__":
    main()
