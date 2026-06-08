#!/usr/bin/env python3
"""Merge per-champion matgrid sweeps; report dual-pass status across the panel."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "phase1" / "meep_research"
DEFAULT_FILES = [
    OUT / "matgrid_calibration_meep_bo_00128.json",
    OUT / "matgrid_calibration_local_00022.json",
    OUT / "matgrid_calibration_meep_bo_00093.json",
]
CHAMPIONS = ["meep_bo_00128", "local_00022", "meep_bo_00093"]
GATE_GAP = 0.03
GATE_DPROD = 0.03


def load_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="*", type=Path, default=DEFAULT_FILES)
    args = p.parse_args()

    merged: list[dict] = []
    for path in args.inputs:
        merged.extend(load_rows(path))

    by_key = {(r["sample_id"], r["label"]): r for r in merged if r.get("status") == "ok"}
    rows = list(by_key.values())

    # Labels that pass on every champion present in merged data
    labels = sorted({r["label"] for r in rows})
    triple: list[tuple[str, dict[str, dict]]] = []
    for lab in labels:
        per = {r["sample_id"]: r for r in rows if r["label"] == lab}
        if not all(s in per for s in CHAMPIONS):
            continue
        if all(
            per[s].get("pass")
            or (
                per[s]["gap"] <= GATE_GAP
                and per[s].get("dprod", 999) <= GATE_DPROD
            )
            for s in CHAMPIONS
        ):
            triple.append((lab, per))

    lines = [
        "# Matgrid calibration — combined panel",
        "",
        f"Gate: mesh gap ≤ {GATE_GAP:.2f}, |r25 − prod| ≤ {GATE_DPROD:.2f}",
        "",
        f"Merged rows: **{len(rows)}** from {len(args.inputs)} file(s).",
        "",
        "## Triple-pass recipes (all 3 champions)",
        "",
    ]
    if triple:
        for lab, per in triple:
            lines.append(f"### `{lab}`")
            lines.append("")
            lines.append("| champion | r25 | r50 | gap | |r25−prod| |")
            lines.append("|----------|-----|-----|-----|------------|")
            for sid in CHAMPIONS:
                r = per[sid]
                lines.append(
                    f"| {sid} | {r['r25']:.3f} | {r['r50']:.3f} | {r['gap']:.3f} | "
                    f"{r.get('dprod', float('nan')):.3f} |"
                )
            lines.append("")
    else:
        lines.append("_None yet — per-champion sweeps in progress._")
        lines.append("")

    lines.append("## Per-champion best dual-pass")
    lines.append("")
    for sid in CHAMPIONS:
        cand = [r for r in rows if r["sample_id"] == sid and r.get("pass")]
        cand.sort(key=lambda r: (r["gap"], r.get("dprod", 0.0)))
        lines.append(f"**{sid}:**")
        if cand:
            b = cand[0]
            lines.append(
                f"- `{b['label']}` gap={b['gap']:.3f} dprod={b.get('dprod', 0):.3f}"
            )
        else:
            lines.append("- _none_")
        lines.append("")

    combined_json = OUT / "matgrid_calibration_combined.json"
    combined_md = OUT / "matgrid_calibration_combined.md"
    combined_json.write_text(json.dumps(rows, indent=2))
    combined_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {combined_json} ({len(rows)} rows)")
    print(f"Wrote {combined_md} ({len(triple)} triple-pass recipe(s))")


if __name__ == "__main__":
    main()
