#!/usr/bin/env python3
"""Config-driven MEEP research panel with promotion-gate scoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nano_inv.crosscheck.meep_backend import run_meep
from nano_inv.crosscheck.types import SolverSpec

OUT = REPO / "data" / "phase1" / "meep_research"
DEFAULT_CFG = REPO / "configs" / "meep_research.yaml"


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

    ref = cfg["reference"]
    champions = cfg["champions"]
    if args.sample_id:
        champions = [c for c in champions if c["id"] == args.sample_id]
        if not champions:
            raise SystemExit(f"unknown sample_id {args.sample_id!r}")

    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    prod_r25: dict[str, float] = {}

    for ch in champions:
        sid = ch["id"]
        mask = np.load(REPO / ch["mask"])
        for exp in cfg["experiments"]:
            label = exp["label"]
            spec = SolverSpec(label, "meep", exp["recipe"], int(exp["res"]), label)
            print(f"==> {sid} / {label}")
            r = run_meep(sid, ch["mask"], mask, spec)
            row = r.to_dict()
            row["recipe_label"] = label
            row["recipe_version"] = exp["recipe"]
            row["resolution"] = int(exp["res"])
            rows.append(row)
            if label == "prod_r25" and r.status == "ok":
                prod_r25[sid] = float(r.split_ratio_upper)
            (OUT / "research.json").write_text(json.dumps(rows, indent=2))

    # Pair r25/r50 by recipe family (strip _r25/_r50 suffix)
    families: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        sid = row["sample_id"]
        label = row["recipe_label"]
        if label.endswith("_r25"):
            fam = label[:-4]
            families.setdefault(fam, {}).setdefault(sid, {})["r25"] = float(row["split_ratio_upper"])
        elif label.endswith("_r50"):
            fam = label[:-4]
            families.setdefault(fam, {}).setdefault(sid, {})["r50"] = float(row["split_ratio_upper"])

    lines = [
        "# MEEP research panel",
        "",
        f"Gate: mesh gap ≤ {max_gap:.2f}, |r25 − prod| ≤ {max_dprod:.2f}",
        "",
        "## Per champion × recipe family",
        "",
        "| sample | family | r25 | r50 | gap | |r25−prod| | pass |",
        "|--------|--------|-----|-----|-----|------------|------|",
    ]

    family_scores: dict[str, list[tuple[float, float]]] = {}
    for fam in sorted(families):
        for sid in sorted(families[fam]):
            splits = families[fam][sid]
            if "r25" not in splits or "r50" not in splits:
                continue
            gap = abs(splits["r50"] - splits["r25"])
            pref = prod_r25.get(sid, float("nan"))
            dprod = abs(splits["r25"] - pref) if np.isfinite(pref) else float("nan")
            ok = gap <= max_gap and dprod <= max_dprod
            family_scores.setdefault(fam, []).append((gap, dprod))
            lines.append(
                f"| {sid} | {fam} | {splits['r25']:.3f} | {splits['r50']:.3f} | "
                f"{gap:.3f} | {dprod:.3f} | {'✓' if ok else '—'} |"
            )

    lines.append("\n## Family summary (max gap, max |r25−prod|)\n")
    ranked: list[tuple[str, float, float, bool]] = []
    for fam, scores in family_scores.items():
        mg = max(g for g, _ in scores)
        md = max(d for _, d in scores)
        dual = mg <= max_gap and md <= max_dprod
        ranked.append((fam, mg, md, dual))
        flag = " **PROMOTE?**" if dual else ""
        lines.append(f"- `{fam}`: max gap **{mg:.3f}**, max |r25−prod| **{md:.3f}**{flag}")

    ranked.sort(key=lambda x: (x[3], x[1], x[2]), reverse=True)
    lines.append("\n## Ranked candidates (dual-pass first)\n")
    for fam, mg, md, dual in ranked:
        lines.append(f"1. `{fam}` — gap {mg:.3f}, dprod {md:.3f}" + (" ✓" if dual else ""))

    (OUT / "research_report.md").write_text("\n".join(lines))
    print(f"Wrote {OUT / 'research_report.md'}")


if __name__ == "__main__":
    main()
