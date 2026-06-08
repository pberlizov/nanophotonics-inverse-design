#!/usr/bin/env python3
"""Export binary mask to GDS (optional gdstk)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mask", type=Path, required=True)
    p.add_argument("--sample-id", type=str, default="design")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/phase1/exports")
    p.add_argument("--pitch-um", type=float, default=4.0, help="Design field width (µm), matches meep design_size_um")
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--datatype", type=int, default=0)
    return p.parse_args()


def write_mask_gds(
    mask: np.ndarray,
    path: Path,
    *,
    pitch_um: float,
    layer: int,
    datatype: int,
) -> None:
    import gdstk

    m = np.asarray(mask)
    if m.ndim > 2:
        m = np.squeeze(m)
    h, w = m.shape
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("TOP")
    dx = pitch_um / w
    dy = pitch_um / h
    for iy in range(h):
        for ix in range(w):
            if m[iy, ix] <= 0.5:
                continue
            x0 = ix * dx
            y0 = iy * dy
            cell.add(
                gdstk.rectangle(
                    (x0, y0),
                    (x0 + dx, y0 + dy),
                    layer=layer,
                    datatype=datatype,
                )
            )
    lib.write_gds(path)


def main() -> None:
    args = parse_args()
    mask_path = args.mask if args.mask.is_absolute() else REPO_ROOT / args.mask
    mask = np.load(mask_path)
    out_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gds_path = out_dir / f"{args.sample_id}.gds"
    try:
        write_mask_gds(
            mask,
            gds_path,
            pitch_um=args.pitch_um,
            layer=args.layer,
            datatype=args.datatype,
        )
    except ImportError:
        raise SystemExit(
            "gdstk not installed. Run: uv pip install gdstk --python .venv/bin/python"
        )

    meta = {
        "sample_id": args.sample_id,
        "mask_path": str(mask_path.relative_to(REPO_ROOT)),
        "gds_path": str(gds_path.relative_to(REPO_ROOT)),
        "pitch_um": args.pitch_um,
        "mask_shape": list(mask.shape),
        "layer": args.layer,
        "datatype": args.datatype,
    }
    (out_dir / f"{args.sample_id}_layout_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
