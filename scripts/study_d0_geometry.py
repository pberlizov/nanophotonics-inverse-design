#!/usr/bin/env python3
"""Phase D0 panel: SDF / analytical-port geometry vs production on champions."""

from __future__ import annotations

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
DEFAULT_CFG = REPO / "configs" / "d0_geometry.yaml"


def load_cfg(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--sample-id", default=None)
    args = p.parse_args()

    cfg = load_cfg(args.config)
    gate = cfg.get("gate") or {}
    max_gap = float(gate.get("max_mesh_gap", 0.03))
    max_dprod = float(gate.get("max_prod_delta", 0.03))

    champions = cfg["champions"]
    if args.sample_id:
        champions = [c for c in champions if c["id"] == args.sample_id]

    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    prod_r25: dict[str, float] = {}

    for ch in champions:
        sid = ch["id"]
        mask = np.load(REPO / ch["mask"])
        for exp in cfg["experiments"]:
            label = exp["label"]
            extra = dict(exp.get("extra") or {})
            base = MeepRecipe.for_version(exp["recipe"], {"resolution": int(exp["res"]), **extra})
            print(f"==> {sid} / {label}")
            r = simulate_mask(mask, base, sample_key=f"{sid}_{label}")
            row = {
                "sample_id": sid,
                "recipe_label": label,
                "recipe_version": exp["recipe"],
                "resolution": int(exp["res"]),
                "status": r.status,
                "split_ratio_upper": float(r.split_ratio_upper),
                "error": r.error,
            }
            rows.append(row)
            if label == "prod_r25" and r.status == "ok":
                prod_r25[sid] = float(r.split_ratio_upper)
            (OUT / "d0_geometry.json").write_text(json.dumps(rows, indent=2))

    # Pair r25/r50 by family prefix
    families: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        sid = row["sample_id"]
        label = row["recipe_label"]
        if label.endswith("_r25"):
            fam = label[:-4]
            families.setdefault(fam, {}).setdefault(sid, {})["r25"] = row["split_ratio_upper"]
        elif label.endswith("_r50"):
            fam = label[:-4]
            families.setdefault(fam, {}).setdefault(sid, {})["r50"] = row["split_ratio_upper"]

    lines = [
        "# Phase D0 geometry panel",
        "",
        f"Gate: mesh gap ≤ {max_gap:.2f}, |r25 − prod| ≤ {max_dprod:.2f}",
        "",
        "| sample | family | r25 | r50 | gap | |r25−prod| | pass |",
        "|--------|--------|-----|-----|-----|------------|------|",
    ]
    ranked: list[tuple[str, float, float, bool]] = []
    for fam in sorted(families):
        max_g = 0.0
        max_d = 0.0
        all_pass = True
        for sid in sorted(families[fam]):
            s = families[fam][sid]
            if "r25" not in s or "r50" not in s:
                continue
            gap = abs(s["r50"] - s["r25"])
            pref = prod_r25.get(sid, float("nan"))
            dprod = abs(s["r25"] - pref) if np.isfinite(pref) else float("nan")
            ok = gap <= max_gap and dprod <= max_dprod
            all_pass = all_pass and ok
            max_g = max(max_g, gap)
            max_d = max(max_d, dprod)
            lines.append(
                f"| {sid} | {fam} | {s['r25']:.3f} | {s['r50']:.3f} | "
                f"{gap:.3f} | {dprod:.3f} | {'✓' if ok else '—'} |"
            )
        ranked.append((fam, max_g, max_d, all_pass))

    ranked.sort(key=lambda x: (not x[3], x[1], x[2]))
    lines.append("\n## Family rank (dual-pass on all champions in panel)\n")
    for fam, mg, md, dual in ranked:
        flag = " **PROMOTE?**" if dual else ""
        lines.append(f"- `{fam}`: max gap **{mg:.3f}**, max |r25−prod| **{md:.3f}**{flag}")

    report = OUT / "d0_geometry_report.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
