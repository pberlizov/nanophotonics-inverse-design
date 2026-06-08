#!/usr/bin/env python3
"""Merge MEEP search / champion rows into the main sim corpus (+ optional manifest patch)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.corpus_merge import (  # noqa: E402
    collect_merge_rows,
    default_champion_sources,
    merge_into_corpus,
)


def patch_manifest(repo_root: Path, manifest_path: Path, new_rows: pd.DataFrame) -> int:
    """Append manifest rows for sample_ids not yet present."""
    if not manifest_path.exists():
        return 0
    man = pd.read_csv(manifest_path)
    existing = set(man["sample_id"].astype(str))
    added = 0
    extra: list[dict] = []
    for _, r in new_rows.iterrows():
        sid = str(r["sample_id"])
        if sid in existing:
            continue
        mp = repo_root / r["mask_path"]
        lp = Path(str(r["mask_path"]).replace("/masks/", "/latents/").replace("_mask.npy", "_latent.npy"))
        lat_col = r.get("latent_path", "")
        if lat_col and (repo_root / str(lat_col)).exists():
            lp = repo_root / str(lat_col)
        if not lp.is_absolute():
            lp = repo_root / lp
        row = {
            "sample_id": sid,
            "source": r.get("source", "meep_search"),
            "latent_path": str(lp.relative_to(repo_root)) if lp.exists() else "",
            "mask_path": str(r["mask_path"]),
            "mask_shape_h": 180,
            "mask_shape_w": 180,
            "drc_heuristic_pass": True,
        }
        if "sigma" in r and pd.notna(r.get("sigma")):
            row["sigma"] = float(r["sigma"])
        extra.append(row)
        added += 1
    if extra:
        man = pd.concat([man, pd.DataFrame(extra)], ignore_index=True)
        man.to_csv(manifest_path, index=False)
    return added


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/corpus_merge.yaml")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    corpus = REPO_ROOT / cfg.get("corpus", "data/phase0/sim_results_phase0_v1_all.csv")
    sources = cfg.get("sources") or default_champion_sources()

    if args.dry_run:
        sources = [
            {**s, "materialize_masks": False} if s.get("kind") == "sim_budget_replicates" else s
            for s in sources
        ]
    new_rows = collect_merge_rows(REPO_ROOT, sources)
    n_before = len(pd.read_csv(corpus)) if corpus.exists() else 0
    summary = {
        "corpus": str(corpus.relative_to(REPO_ROOT)),
        "n_new_rows_collected": len(new_rows),
        "n_before": n_before,
        "new_sample_ids": sorted(new_rows["sample_id"].astype(str).unique().tolist()) if len(new_rows) else [],
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return

    if len(new_rows) == 0:
        print("no rows to merge")
        return

    merged = merge_into_corpus(
        REPO_ROOT, corpus, new_rows, backup=bool(cfg.get("backup", True))
    )
    summary["n_after"] = len(merged)
    summary["n_added_or_updated"] = summary["n_after"] - n_before

    if cfg.get("patch_manifest", True):
        manifest = REPO_ROOT / cfg.get("manifest", "data/phase0/manifest.csv")
        summary["manifest_rows_added"] = patch_manifest(REPO_ROOT, manifest, new_rows)

    out = REPO_ROOT / "data/phase1/corpus_merge_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
