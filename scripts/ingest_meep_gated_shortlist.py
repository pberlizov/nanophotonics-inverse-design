#!/usr/bin/env python3
"""Append MEEP-gated shortlist results to corpus + manifest.

  PYTHONPATH=src python scripts/ingest_meep_gated_shortlist.py
  PYTHONPATH=src python scripts/ingest_meep_gated_shortlist.py --meep-csv data/phase1/wedge_a/meep_gated_shortlist/meep_prod_r25_top15.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nano_inv.manifest import append_sim_results

import json

DEFAULT_MEEP = REPO / "data/phase1/wedge_a/meep_gated_shortlist/meep_prod_r25_top15.csv"
DEFAULT_RANKED = REPO / "data/phase1/wedge_a/meep_gated_shortlist/ranked_500.csv"
CORPUS = REPO / "data/phase0/sim_results_phase0_v1_all.csv"
MANIFEST = REPO / "data/phase0/manifest.csv"
D0_GEOM = REPO / "data/phase1/meep_research/d0_geometry.json"
PANEL_JSON = REPO / "data/phase1/wedge_a/meep_gated_shortlist/promotion_panel_cand_000160.json"


def patch_manifest(rows: pd.DataFrame) -> int:
    if not MANIFEST.exists():
        return 0
    man = pd.read_csv(MANIFEST)
    existing = set(man["sample_id"].astype(str))
    extra: list[dict] = []
    for _, r in rows.iterrows():
        sid = str(r["sample_id"])
        if sid in existing:
            continue
        latent = r.get("latent_path", "")
        mask = r["mask_path"]
        extra.append(
            {
                "sample_id": sid,
                "source": r.get("source", "meep_gated_shortlist"),
                "latent_path": latent,
                "mask_path": mask,
                "mask_shape_h": 180,
                "mask_shape_w": 180,
                "drc_heuristic_pass": True,
                "sigma": r.get("sigma"),
            }
        )
    if not extra:
        return 0
    pd.concat([man, pd.DataFrame(extra)], ignore_index=True).to_csv(MANIFEST, index=False)
    return len(extra)


def append_d0_from_panel(sample_id: str = "cand_000160") -> None:
    """Add prod + sdf_geom rows so promote_meep_recipe --skip-existing skips re-sim."""
    if not PANEL_JSON.exists() or not D0_GEOM.exists():
        return
    panel = {r["sample_id"]: r for r in json.loads(PANEL_JSON.read_text())}
    if sample_id not in panel:
        return
    rows = json.loads(D0_GEOM.read_text())
    existing = {(r["sample_id"], r.get("recipe_label")) for r in rows}
    ent = panel[sample_id]
    for m in ent["meep"]:
        lab = m["experiment"]
        if lab == "prod_r25":
            label, ver = "prod_r25", "phase0_v1"
        elif lab.startswith("sdf_geom"):
            label, ver = lab, "phase0_v1_sdf_geom"
        else:
            continue
        key = (sample_id, label)
        if key in existing:
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "recipe_label": label,
                "recipe_version": ver,
                "resolution": 50 if "r50" in label else 25,
                "status": m["status"],
                "split_ratio_upper": m["split_ratio_upper"],
                "error": "",
            }
        )
    D0_GEOM.write_text(json.dumps(rows, indent=2))
    print(f"Updated {D0_GEOM} with {sample_id} panel rows")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meep-csv", type=Path, default=DEFAULT_MEEP)
    p.add_argument("--ranked-csv", type=Path, default=DEFAULT_RANKED)
    p.add_argument("--corpus", type=Path, default=CORPUS)
    p.add_argument("--source-tag", default="meep_gated_shortlist")
    p.add_argument("--append-d0-panel", action="store_true", default=True)
    p.add_argument("--no-append-d0-panel", action="store_false", dest="append_d0_panel")
    args = p.parse_args()

    meep = pd.read_csv(args.meep_csv)
    meep = meep[meep["status"] == "ok"].copy()
    meep["source"] = args.source_tag

    ranked = pd.read_csv(args.ranked_csv)
    lat_map = dict(zip(ranked["sample_id"].astype(str), ranked["latent_path"].astype(str)))
    sig_map = dict(zip(ranked["sample_id"].astype(str), ranked.get("sigma", pd.Series(dtype=float))))

    manifest_rows = []
    for _, row in meep.iterrows():
        sid = str(row["sample_id"])
        manifest_rows.append(
            {
                "sample_id": sid,
                "source": args.source_tag,
                "mask_path": row["mask_path"],
                "latent_path": lat_map.get(sid, ""),
                "sigma": sig_map.get(sid),
            }
        )

    combined = append_sim_results(args.corpus, meep)
    n_man = patch_manifest(pd.DataFrame(manifest_rows))
    print(f"Corpus rows (total): {len(combined)} (ingested {len(meep)} MEEP ok rows)")
    print(f"Manifest rows added: {n_man}")
    if args.append_d0_panel:
        append_d0_from_panel()


if __name__ == "__main__":
    main()
