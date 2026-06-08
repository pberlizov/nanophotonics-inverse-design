#!/usr/bin/env python3
"""Product path: STE surrogate refine → verify with prod r25 gate.

Pass criteria (default): |split - 0.5| < 0.03 on prod r25 (refined) and mask Δ < 10%.

  PYTHONPATH=src python scripts/refine_champion_product.py --sample-id local_00022
  PYTHONPATH=src python scripts/refine_champion_product.py --sample-id local_00022 --meep
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = REPO / ".venv/bin/python"


def prod_gate(report: list[dict], *, split_tol: float, mask_tol: float) -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok_all = True
    for row in report:
        sid = row["sample_id"]
        mask_frac = row.get("mask_frac_changed")
        if mask_frac is not None and mask_frac > mask_tol:
            ok_all = False
            notes.append(f"{sid}: mask Δ {mask_frac*100:.1f}% > {mask_tol*100:.0f}%")
        prod = (row.get("meep_delta") or {}).get("prod_r25")
        if prod is None:
            ref_split = next(
                (
                    m["split_ratio_upper"]
                    for m in row["variants"]["refined"].get("meep", [])
                    if m.get("experiment") == "prod_r25" and m.get("status") == "ok"
                ),
                None,
            )
            if ref_split is None:
                ok_all = False
                notes.append(f"{sid}: missing prod_r25 MEEP")
                continue
        else:
            ref_split = prod[0]
        if ref_split is None:
            ok_all = False
            notes.append(f"{sid}: prod_r25 failed")
            continue
        err = abs(ref_split - 0.5)
        if err > split_tol:
            ok_all = False
            notes.append(f"{sid}: prod r25 split={ref_split:.4f} (|err|={err:.4f} > {split_tol})")
        else:
            notes.append(f"{sid}: PASS prod r25={ref_split:.4f}, mask Δ={(mask_frac or 0)*100:.1f}%")
    return ok_all, notes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO / "configs/refine_champion_surrogate.yaml")
    p.add_argument("--sample-id", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--skip-refine", action="store_true")
    p.add_argument("--gym-only", action="store_true")
    p.add_argument("--meep", action="store_true", help="Run MEEP verify after gym (uses run_meep.sh)")
    p.add_argument("--split-tol", type=float, default=0.03)
    p.add_argument("--mask-tol", type=float, default=0.10)
    p.add_argument("--refine-dir", type=Path, default=REPO / "data/phase1/refine_surrogate_ste")
    p.add_argument(
        "--refine-when-off-target-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip STE refine when corpus prod r25 already |split−0.5| < prod-split-tol (default: on)",
    )
    p.add_argument("--prod-split-tol", type=float, default=0.03)
    p.add_argument("--prod-penalty-weight", type=float, default=0.0)
    p.add_argument(
        "--force-refine",
        action="store_true",
        help="Run STE refine even when corpus prod split is on-target",
    )
    args = p.parse_args()

    refine_cmd = [
        str(PY) if PY.exists() else sys.executable,
        str(REPO / "scripts/refine_champion_surrogate.py"),
        "--config",
        str(args.config),
        "--output-dir",
        str(args.refine_dir),
    ]
    if args.sample_id:
        refine_cmd.extend(["--sample-id", args.sample_id])
    if args.steps is not None:
        refine_cmd.extend(["--steps", str(args.steps)])
    if args.refine_when_off_target_only:
        refine_cmd.append("--refine-when-off-target-only")
    else:
        refine_cmd.append("--no-refine-when-off-target-only")
    refine_cmd.extend(["--prod-split-tol", str(args.prod_split_tol)])
    if args.prod_penalty_weight > 0:
        refine_cmd.extend(["--prod-penalty-weight", str(args.prod_penalty_weight)])
    if args.force_refine:
        refine_cmd.append("--force-refine")

    if not args.skip_refine:
        print("+", " ".join(refine_cmd))
        subprocess.run(
            refine_cmd, cwd=REPO, check=True, env={**os.environ, "PYTHONPATH": "src"}
        )

    refine_json = args.refine_dir / "refine_champion_surrogate.json"
    verify_cmd = [
        str(PY) if PY.exists() else sys.executable,
        str(REPO / "scripts/verify_refined_champions.py"),
        "--refine-source",
        "surrogate",
        "--refine-json",
        str(refine_json),
        "--refine-dir",
        str(args.refine_dir),
        "--latent-suffix",
        "surrogate_ste",
    ]
    if args.sample_id:
        verify_cmd.extend(["--sample-id", args.sample_id])
    # MEEP runs only via run_meep.sh (conda mp); verify step is always gym-only here.
    verify_cmd.append("--gym-only")

    env = {**os.environ, "PYTHONPATH": "src"}
    print("+", " ".join(verify_cmd))
    subprocess.run(verify_cmd, cwd=REPO, check=True, env=env)

    if args.meep:
        meep_cmd = [
            "bash",
            str(REPO / "scripts/run_meep.sh"),
            str(REPO / "scripts/verify_refined_champions.py"),
            "--meep-only",
            "--refine-source",
            "surrogate",
            "--refine-json",
            str(refine_json),
            "--refine-dir",
            str(args.refine_dir),
            "--latent-suffix",
            "surrogate_ste",
        ]
        if args.sample_id:
            meep_cmd.extend(["--sample-id", args.sample_id])
        print("+", " ".join(meep_cmd))
        subprocess.run(meep_cmd, cwd=REPO, check=True)

        report_cmd = verify_cmd[:-1] + ["--report-only"]
        subprocess.run(report_cmd, cwd=REPO, check=True, env=env)

    verify_json = args.refine_dir / "verify_refined_surrogate_ste.json"
    if not verify_json.exists():
        print(f"No verify report at {verify_json}", file=sys.stderr)
        sys.exit(1)
    report = json.loads(verify_json.read_text())
    passed, notes = prod_gate(report, split_tol=args.split_tol, mask_tol=args.mask_tol)
    for line in notes:
        print(line)
    gate_path = args.refine_dir / "product_gate.json"
    gate_path.write_text(
        json.dumps(
            {"passed": passed, "split_tol": args.split_tol, "mask_tol": args.mask_tol, "notes": notes},
            indent=2,
        )
    )
    print(f"PRODUCT GATE: {'PASS' if passed else 'FAIL'} (wrote {gate_path})")
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
