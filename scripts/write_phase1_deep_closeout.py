#!/usr/bin/env python3
"""Aggregate Phase 1 deep-dev artifacts into one JSON closeout."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def main() -> None:
    closeout = {
        "champion_local": _load(REPO / "data/phase1/meep_search_local/meep_search_local_summary.json"),
        "champion_local_deep": _load(REPO / "data/phase1/meep_search_local_deep/meep_search_local_summary.json"),
        "meep_search_deep": _load(REPO / "data/phase1/meep_search_deep/meep_search_summary.json"),
        "al_round_01": _load(REPO / "data/phase1/al_round_01/round_summary.json"),
        "ranking_eval": _load(REPO / "data/phase1/al_round_01/ranking_eval.json"),
        "resim_meep_bo_00128": None,
    }
    import pandas as pd

    resim_frames = []
    for path in sorted((REPO / "data/phase1").glob("meep_bo_00128_resim_*.csv")):
        resim_frames.append(pd.read_csv(path))
    if resim_frames:
        df = pd.concat(resim_frames, ignore_index=True)
        ok = df[df["status"] == "ok"]
        if len(ok):
            closeout["resim_meep_bo_00128"] = {
                "n_runs": len(ok),
                "split_mean": float(ok["split_ratio_upper"].mean()),
                "split_std": float(ok["split_ratio_upper"].std()),
                "rows": ok[["sample_id", "split_ratio_upper", "insertion_loss_db"]].to_dict("records"),
            }

    out = REPO / "data/phase1/phase1_deep_closeout.json"
    out.write_text(json.dumps(closeout, indent=2))
    print(json.dumps(closeout, indent=2))


if __name__ == "__main__":
    main()
