#!/usr/bin/env python3
"""Sweep matgrid geometry knobs; score r25/r50 gap vs production offset."""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nano_inv.meep_sim import MeepRecipe, simulate_mask

OUT = REPO / "data" / "phase1" / "meep_research"
RESULT_PATH = OUT / "matgrid_calibration.json"
VALIDATION_PATH = OUT / "matgrid_validation.json"
DEFAULT_CFG = REPO / "configs" / "matgrid_calibration.yaml"


def load_cfg(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def recipe_for(overrides: dict, resolution: int) -> MeepRecipe:
    version = "phase0_v1_matgrid_avg" if overrides.get("eps_averaging") else "phase0_v1_matgrid"
    base = MeepRecipe.for_version(version, {"resolution": resolution})
    arm_y = overrides.get("arm_y")
    fields = {
        k: overrides[k]
        for k in (
            "wg_width_um",
            "arm_half_height_um",
            "port_overlap_um",
            "matgrid_upsample",
        )
        if k in overrides
    }
    if arm_y is not None:
        fields["arm_y_upper"] = float(arm_y)
        fields["arm_y_lower"] = -float(arm_y)
    return replace(base, **fields)


def combo_label(overrides: dict) -> str:
    parts = [
        f"wg={overrides['wg_width_um']:.2f}",
        f"arm={overrides['arm_y']:.2f}",
        f"ah={overrides['arm_half_height_um']:.2f}",
        f"ov={overrides['port_overlap_um']:.2f}",
        f"avg={'1' if overrides['eps_averaging'] else '0'}",
        f"up={overrides['matgrid_upsample']}",
    ]
    return "_".join(parts)


def load_result_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def paths_for_cfg(cfg: dict) -> tuple[Path, Path]:
    tag = cfg.get("output_tag")
    if tag:
        return OUT / f"matgrid_calibration_{tag}.json", OUT / f"matgrid_calibration_{tag}.md"
    return RESULT_PATH, OUT / "matgrid_calibration.md"


def save_result_rows(path: Path, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))


def merge_rows(existing: list[dict], new_row: dict) -> list[dict]:
    by_key = {(r["sample_id"], r["label"]): r for r in existing}
    by_key[(new_row["sample_id"], new_row["label"])] = new_row
    return list(by_key.values())


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--sample-id", default=None, help="Champion id (default: first in config)")
    p.add_argument("--all-champions", action="store_true", help="Run full sweep on every champion")
    p.add_argument("--top", type=int, default=5, help="Re-validate top N combos on all champions")
    p.add_argument("--validate-only", type=Path, default=None, help="JSON from prior sweep")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip combos already marked ok in matgrid_calibration.json",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore saved results and rerun the full sweep",
    )
    args = p.parse_args()

    cfg = load_cfg(args.config)
    result_path, report_path = paths_for_cfg(cfg)
    gate = cfg.get("gate") or {}
    max_gap = float(gate.get("max_mesh_gap", 0.03))
    max_dprod = float(gate.get("max_prod_delta", 0.03))
    resolutions = [int(r) for r in cfg.get("resolutions", [25, 50])]

    champions = cfg["champions"]
    if args.sample_id:
        champions = [c for c in champions if c["id"] == args.sample_id]
        if not champions:
            raise SystemExit(f"unknown sample_id {args.sample_id!r}")

    OUT.mkdir(parents=True, exist_ok=True)

    def prod_r25(sid: str, mask: np.ndarray) -> float:
        r = simulate_mask(
            mask,
            MeepRecipe.for_version("phase0_v1", {"resolution": 25}),
            sample_key=f"{sid}_prod",
        )
        if r.status != "ok":
            raise RuntimeError(f"production r25 failed for {sid}: {r.error}")
        return float(r.split_ratio_upper)

    def run_combo(sid: str, mask: np.ndarray, overrides: dict) -> dict:
        label = combo_label(overrides)
        splits: dict[int, float] = {}
        for res in resolutions:
            recipe = recipe_for(overrides, res)
            print(f"  {sid} / {label} @ r{res}")
            r = simulate_mask(mask, recipe, sample_key=f"{sid}_{label}_r{res}")
            if r.status != "ok":
                return {
                    "sample_id": sid,
                    "label": label,
                    "overrides": overrides,
                    "status": "error",
                    "error": r.error,
                }
            splits[res] = float(r.split_ratio_upper)
        r25, r50 = splits[resolutions[0]], splits[resolutions[-1]]
        gap = abs(r50 - r25)
        return {
            "sample_id": sid,
            "label": label,
            "overrides": overrides,
            "status": "ok",
            "r25": r25,
            "r50": r50,
            "gap": gap,
            "splits": {str(k): v for k, v in splits.items()},
        }

    resume = args.resume or (result_path.is_file() and not args.fresh and not args.validate_only)
    rows: list[dict] = [] if args.fresh else load_result_rows(result_path)
    done = {
        (r["sample_id"], r["label"])
        for r in rows
        if r.get("status") == "ok"
    }

    if args.validate_only:
        prior = json.loads(args.validate_only.read_text())
        ref_sid = cfg.get("output_tag") or (champions[0]["id"] if champions else "meep_bo_00128")
        sweep_rows = [r for r in prior if r.get("status") == "ok" and r["sample_id"] == ref_sid]
        if not sweep_rows:
            sweep_rows = [r for r in prior if r.get("status") == "ok"]
            ref_sid = sweep_rows[0]["sample_id"] if sweep_rows else ref_sid
        sweep_rows.sort(key=lambda r: (not r.get("pass", False), r["gap"], r.get("dprod", 0.0)))
        seen: set[str] = set()
        combos: list[dict] = []
        for r in sweep_rows:
            lab = r["label"]
            if lab in seen:
                continue
            seen.add(lab)
            combos.append(r["overrides"])
            if len(combos) >= args.top:
                break
        print(f"Validating top {len(combos)} combos on all champions (from {ref_sid} sweep)")
        rows = load_result_rows(VALIDATION_PATH)
        prod: dict[str, float] = {}
        for ch in cfg["champions"]:
            sid = ch["id"]
            mask = np.load(REPO / ch["mask"])
            prod[sid] = prod_r25(sid, mask)
        for overrides in combos:
            for ch in cfg["champions"]:
                sid = ch["id"]
                mask = np.load(REPO / ch["mask"])
                row = run_combo(sid, mask, overrides)
                if row["status"] == "ok":
                    row["dprod"] = abs(row["r25"] - prod[sid])
                    row["pass"] = row["gap"] <= max_gap and row["dprod"] <= max_dprod
                rows = merge_rows(rows, row)
                VALIDATION_PATH.write_text(json.dumps(rows, indent=2))
    else:
        sweep = cfg["sweep"]
        keys = list(sweep.keys())
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*(sweep[k] for k in keys))]
        target_champs = cfg["champions"] if args.all_champions else champions[:1]
        pending = sum(
            1
            for ch in target_champs
            for overrides in combos
            if (ch["id"], combo_label(overrides)) not in done
        )
        print(
            f"Sweep: {len(combos)} combos × {len(target_champs)} champion(s)"
            + (f" ({len(done)} done, {pending} pending)" if resume else "")
        )

        prod: dict[str, float] = {}
        for ch in target_champs:
            sid = ch["id"]
            mask = np.load(REPO / ch["mask"])
            prod[sid] = prod_r25(sid, mask)
            for overrides in combos:
                label = combo_label(overrides)
                if resume and (sid, label) in done:
                    print(f"  skip {sid} / {label} (done)")
                    continue
                row = run_combo(sid, mask, overrides)
                if row["status"] == "ok":
                    row["dprod"] = abs(row["r25"] - prod[sid])
                    row["pass"] = row["gap"] <= max_gap and row["dprod"] <= max_dprod
                rows = merge_rows(rows, row)
                save_result_rows(result_path, rows)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    ok_rows.sort(key=lambda r: (not r.get("pass", False), r["gap"], r.get("dprod", 0.0)))

    lines = [
        "# Matgrid geometry calibration",
        "",
        f"Gate: mesh gap ≤ {max_gap:.2f}, |r25 − prod| ≤ {max_dprod:.2f}",
        "",
        "## Ranked results",
        "",
        "| sample | label | r25 | r50 | gap | |r25−prod| | pass |",
        "|--------|-------|-----|-----|-----|------------|------|",
    ]
    for r in ok_rows[:40]:
        lines.append(
            f"| {r['sample_id']} | `{r['label']}` | {r['r25']:.3f} | {r['r50']:.3f} | "
            f"{r['gap']:.3f} | {r.get('dprod', float('nan')):.3f} | "
            f"{'✓' if r.get('pass') else '—'} |"
        )

    dual = [r for r in ok_rows if r.get("pass")]
    lines.append("\n## Dual-pass count\n")
    lines.append(f"{len(dual)} / {len(ok_rows)} rows pass the gate.")
    if dual:
        lines.append("\nBest dual-pass:\n")
        for r in dual[:5]:
            lines.append(f"- `{r['label']}` on **{r['sample_id']}** (gap={r['gap']:.3f})")

    report = VALIDATION_PATH.with_suffix(".md") if args.validate_only else report_path
    report.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
