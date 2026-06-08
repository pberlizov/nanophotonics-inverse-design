#!/usr/bin/env python3
"""
Cross-check champion masks on several solvers (MEEP template family + Tidy3D + scalar BPM).

Usage:
  # MEEP backends (conda env mp):
  bash scripts/run_meep.sh scripts/crosscheck_champion_solvers.py --solvers meep

  # Tidy3D + BPM (.venv):
  PYTHONPATH=src .venv/bin/python scripts/crosscheck_champion_solvers.py --solvers tidy3d,bpm --append

  # All (wrapper):
  bash scripts/run_crosscheck.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.crosscheck.bpm_backend import run_bpm
from nano_inv.crosscheck.meep_backend import run_meep
from nano_inv.crosscheck.tidy3d_backend import run_tidy3d
from nano_inv.crosscheck.types import CrosscheckResult, SolverSpec

OUT_DIR = REPO_ROOT / "data/phase1/crosscheck"

# Tidy3D credits: run these three only (skip ref — not 50/50; saves ~1/4 budget).
TIDY3D_CHAMPION_IDS = {"local_00022", "meep_bo_00128", "meep_bo_00093"}

CHAMPIONS = [
    {
        "sample_id": "ref_published",
        "mask_path": "data/phase0/masks/ref_published_mask.npy",
        "expected_split": 0.614,
    },
    {
        "sample_id": "local_00022",
        "mask_path": "data/phase1/meep_search_local/candidates/masks/local_00022_mask.npy",
        "expected_split": 0.500,
    },
    {
        "sample_id": "meep_bo_00128",
        "mask_path": "data/phase1/meep_search_deep/candidates/masks/meep_bo_00128_mask.npy",
        "expected_split": 0.509,
    },
    {
        "sample_id": "meep_bo_00093",
        "mask_path": "data/phase0/meep_search_100/candidates/masks/meep_bo_00093_mask.npy",
        "expected_split": 0.497,
    },
]

MEEP_SOLVERS = [
    SolverSpec("meep_phase0_v1_r25", "meep", "phase0_v1", 25, "Production template (reference)"),
    SolverSpec("meep_phase0_v1_r50", "meep", "phase0_v1", 50, "Resolution convergence check"),
    SolverSpec(
        "meep_phase0_v1_sdf_geom_r25",
        "meep",
        "phase0_v1_sdf_geom",
        25,
        "Mesh-stable audit (promoted)",
    ),
    SolverSpec(
        "meep_phase0_v1_sdf_geom_r50",
        "meep",
        "phase0_v1_sdf_geom",
        50,
        "Mesh-stable r50 confirmation",
    ),
    SolverSpec("meep_phase0_v0_r25", "meep", "phase0_v0", 25, "Legacy recipe (no flux decay)"),
    SolverSpec("meep_phase0_v2_r25", "meep", "phase0_v2", 25, "Template v2 (flip_y + arm spacing)"),
    SolverSpec("meep_phase0_v1_r15", "meep", "phase0_v1", 15, "Fast MEEP (coarse grid)"),
]

ALT_SOLVERS = [
    SolverSpec("tidy3d_phase0_v1", "tidy3d", "phase0_v1", 25, "Tidy3D cloud FDTD (requires API key)"),
    SolverSpec("bpm_scalar_phase0_v1", "bpm", "phase0_v1", 25, "Scalar BPM — qualitative only (optional)"),
]

REFERENCE_SOLVER = "meep_phase0_v1_r25"


def _load_existing(path: Path) -> list[dict]:
    if path.exists():
        data = json.loads(path.read_text())
        return list(data.get("results", []))
    return []


def _merge_results(existing: list[dict], new_rows: list[CrosscheckResult]) -> list[dict]:
    by_key = {(r["sample_id"], r["solver"]): r for r in existing}
    for row in new_rows:
        by_key[(row.sample_id, row.solver)] = row.to_dict()
    return list(by_key.values())


def _reference_splits(results: list[dict]) -> dict[str, float]:
    ref = {}
    for r in results:
        if r.get("solver") == REFERENCE_SOLVER and r.get("status") == "ok":
            ref[r["sample_id"]] = float(r["split_ratio_upper"])
    if ref:
        return ref
    # Fallback: corpus MEEP phase0_v1 r25 (when only Tidy3D rows in this run).
    import pandas as pd

    sim_path = REPO_ROOT / "data/phase0/sim_results_phase0_v1_all.csv"
    if sim_path.exists():
        df = pd.read_csv(sim_path)
        for ch in CHAMPIONS:
            sid = ch["sample_id"]
            sub = df[(df["sample_id"] == sid) & (df.get("recipe_version", "phase0_v1") == "phase0_v1")]
            if len(sub) and sub.iloc[0].get("status") == "ok":
                ref[sid] = float(sub.iloc[0]["split_ratio_upper"])
    return ref


def write_report(results: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_splits = _reference_splits(results)

    solvers = sorted({r["solver"] for r in results})
    lines = [
        "# Champion mask — multi-solver cross-check",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "Shared geometry: 6 µm cell, 4 µm design, C-band (1.55 µm), drcgenerator masks.",
        f"Reference for |Δsplit|: **{REFERENCE_SOLVER}** (MEEP `phase0_v1`, res 25).",
        "",
        "## MEEP vs Tidy3D (production contract)",
        "",
        "| sample_id | MEEP r25 | Tidy3D | |Δ| | In-spec both? |",
        "|-----------|----------|--------|-----|---------------|",
    ]
    for ch in CHAMPIONS:
        sid = ch["sample_id"]
        meep_s = ref_splits.get(sid)
        td_row = next((r for r in results if r["sample_id"] == sid and "tidy3d" in r["solver"]), None)
        td_s = float(td_row["split_ratio_upper"]) if td_row and td_row.get("status") == "ok" else float("nan")
        if meep_s is not None and np.isfinite(td_s):
            d = abs(td_s - meep_s)
            inspec = abs(meep_s - 0.5) <= 0.05 and abs(td_s - 0.5) <= 0.05
            lines.append(
                f"| {sid} | {meep_s:.3f} | {td_s:.3f} | {d:.3f} | {'yes' if inspec else 'no'} |"
            )
        elif np.isfinite(td_s):
            lines.append(f"| {sid} | — | {td_s:.3f} | — | — |")
    lines.extend(
        [
            "",
            "## Split ratio (upper arm) — all solvers",
            "",
            "| sample_id | " + " | ".join(solvers) + " |",
            "|---|" + "|".join(["---"] * len(solvers)) + "|",
        ]
    )

    solvers = sorted({r["solver"] for r in results})
    for ch in CHAMPIONS:
        sid = ch["sample_id"]
        cells = [sid]
        for solver in solvers:
            row = next((r for r in results if r["sample_id"] == sid and r["solver"] == solver), None)
            if not row or row.get("status") != "ok":
                cells.append("—")
                continue
            split = row["split_ratio_upper"]
            ref = ref_splits.get(sid)
            err = abs(split - ref) if ref is not None and solver != REFERENCE_SOLVER else 0.0
            if solver == REFERENCE_SOLVER:
                cells.append(f"**{split:.3f}**")
            elif np.isfinite(err):
                cells.append(f"{split:.3f} (Δ{err:.3f})")
            else:
                cells.append(f"{split:.3f}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- **MEEP** is the production solver (`SIM_CONTRACT.md`).",
            "- **Tidy3D** (optional): cloud FDTD via `tidy3d.web.run` when API key is configured; same rasterized ε grid.",
            "- **BPM** is a fast scalar paraxial check — trends only, not for IL or absolute split claims.",
            "- **Lumerical / COMSOL**: no automated license in this repo. Import GDS/mask manually; compare split @ 1550 nm.",
            "",
            "## Outreach positioning",
            "",
            "- Production `phase0_v1` uses `mask_flip_y: true` (see `configs/phase0.yaml`).",
            "- Champions should agree between MEEP r25 and r50 within ~0.05 split unless mesh-limited.",
            "- Cross-solver agreement supports *sim-qualified* claims, not fab validation.",
            "",
        ]
    )

    (out_dir / "crosscheck_report.md").write_text("\n".join(lines))

    import csv

    csv_path = out_dir / "crosscheck_results.csv"
    fields = [
        "sample_id",
        "mask_path",
        "solver",
        "status",
        "split_ratio_upper",
        "insertion_loss_db",
        "abs_err_vs_reference",
        "flux_in",
        "flux_out_upper",
        "flux_out_lower",
        "error",
        "runtime_note",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(results, key=lambda x: (x["sample_id"], x["solver"])):
            w.writerow(r)

    lumerical = out_dir / "LUMERICAL_MANUAL.md"
    lumerical.write_text(
        """# Lumerical manual cross-check (offline)

No Lumerical API license is configured in this repo. To compare champions manually:

1. Export mask: `*.npy` → PNG/GDS (see `scripts/package_pilot_outreach.py` GDS path).
2. Build the same layout in FDTD Solutions:
   - Cell 6×6 µm, design region 4×4 µm centered.
   - Si ε=12, SiO₂ ε=2.25; wg width 0.45 µm; output arms at y=±0.6 µm.
   - TE-like excitation at x ≈ -2.0 µm, monitors at upper/lower arm flux planes.
3. Record split = P_upper / (P_upper + P_lower) at λ = 1.55 µm.
4. Compare to `crosscheck_results.csv` column `meep_phase0_v1_r25`.

Expect differences from subpixel smoothing, PML, and source placement vs our MEEP template.
"""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-solver champion cross-check")
    parser.add_argument(
        "--solvers",
        default="meep,tidy3d,bpm",
        help="Comma-separated: meep, tidy3d, bpm",
    )
    parser.add_argument("--append", action="store_true", help="Merge with existing JSON")
    parser.add_argument("--limit", type=int, default=0, help="Max champions (0 = all)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--tidy3d-frugal",
        action="store_true",
        help="Lower grid/runtime for Tidy3D (saves FlexCredits)",
    )
    parser.add_argument(
        "--tidy3d-estimate",
        action="store_true",
        help="Only estimate FlexCredit cost per sim (no upload)",
    )
    parser.add_argument(
        "--tidy3d-include-ref",
        action="store_true",
        help="Also run Tidy3D on ref_published (uses extra credits)",
    )
    args = parser.parse_args()

    families = {s.strip().lower() for s in args.solvers.split(",")}
    champions = CHAMPIONS[: args.limit] if args.limit else CHAMPIONS

    json_path = OUT_DIR / "crosscheck_report.json"
    existing = _load_existing(json_path) if args.append else []
    ref_splits = _reference_splits(existing)

    new_results: list[CrosscheckResult] = []

    for ch in champions:
        mask_path = REPO_ROOT / ch["mask_path"]
        if not mask_path.exists():
            print(f"SKIP missing mask: {mask_path}")
            continue
        mask = np.load(mask_path)
        sid = ch["sample_id"]
        ref = ref_splits.get(sid)

        if "meep" in families:
            for spec in MEEP_SOLVERS:
                print(f"==> {sid} / {spec.name}")
                row = run_meep(sid, str(mask_path.relative_to(REPO_ROOT)), mask, spec, reference_split=ref, verbose=args.verbose)
                new_results.append(row)
                if spec.name == REFERENCE_SOLVER and row.status == "ok":
                    ref = row.split_ratio_upper
                    ref_splits[sid] = ref

        if "tidy3d" in families:
            if not args.tidy3d_include_ref and sid not in TIDY3D_CHAMPION_IDS:
                print(f"SKIP {sid} (tidy3d-champions-only)")
                continue
            spec = ALT_SOLVERS[0]
            print(f"==> {sid} / {spec.name}")
            frugal = args.tidy3d_frugal
            new_results.append(
                run_tidy3d(
                    sid,
                    str(mask_path.relative_to(REPO_ROOT)),
                    mask,
                    spec,
                    reference_split=ref,
                    run_time_ps=2.5 if frugal else 4.0,
                    min_steps_per_wvl=10 if frugal else 12,
                    estimate_only=args.tidy3d_estimate,
                )
            )

        if "bpm" in families:
            spec = ALT_SOLVERS[1]
            print(f"==> {sid} / {spec.name}")
            new_results.append(
                run_bpm(sid, str(mask_path.relative_to(REPO_ROOT)), mask, spec, reference_split=ref)
            )

    merged = _merge_results(existing, new_results)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reference_solver": REFERENCE_SOLVER,
        "champions": CHAMPIONS,
        "results": merged,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2))
    write_report(merged, OUT_DIR)
    print(f"Wrote {json_path}")
    print(f"Wrote {OUT_DIR / 'crosscheck_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
