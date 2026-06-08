"""Manifest helpers for active learning and training merges."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def filter_by_source(
    table: pd.DataFrame,
    source_filter: str | list[str] | None,
) -> pd.DataFrame:
    """Filter manifest rows by source column (no sklearn/joblib deps)."""
    if source_filter is None or source_filter == "all":
        return table
    if isinstance(source_filter, str):
        if source_filter == "perturb":
            allowed = {"perturbation", "published_reference"}
        elif source_filter == "perturb_plus_search":
            allowed = {
                "perturbation",
                "published_reference",
                "meep_search_local",
                "meep_search_deep",
                "meep_search_phase0",
                "meep_search",
                "sim_budget",
                "meep_gated_shortlist",
                "meep_gated_shortlist_r2",
                "meep_gated_shortlist_r3",
                "meep_gated_shortlist_r4",
                "meep_gated_shortlist_r5",
                "meep_gated_shortlist_r6",
                "meep_gated_shortlist_r7",
                "meep_gated_shortlist_rank",
            }
        elif source_filter == "perlin":
            allowed = {"perlin"}
        else:
            allowed = {source_filter}
    else:
        allowed = set(source_filter)
    col = "source" if "source" in table.columns else None
    if col is None:
        for c in table.columns:
            if c.startswith("source"):
                col = c
                break
    if col is None:
        return table
    return table.loc[table[col].isin(allowed)].reset_index(drop=True)


def build_al_training_manifest(
    repo_root: Path,
    base_manifest: Path,
    candidate_csv: Path,
    out_path: Path,
    *,
    base_source_filter: str = "perturb",
) -> pd.DataFrame:
    """
    Baseline manifest rows (e.g. perturb-only) plus search candidates for retrain.

    Search rows use source=search_bo so they are kept when training with --sources all
    on this file only.
    """
    base = pd.read_csv(base_manifest)
    base = filter_by_source(base, base_source_filter)

    cand = pd.read_csv(candidate_csv)
    search_rows: list[dict] = []
    for _, r in cand.iterrows():
        search_rows.append(
            {
                "sample_id": r["sample_id"],
                "source": "search_bo",
                "latent_path": r["latent_path"],
                "mask_path": r["mask_path"],
                "mask_shape_h": 180,
                "mask_shape_w": 180,
                "drc_heuristic_pass": r.get("drc_heuristic_pass", True),
                "drc_reasons": "",
                "family": r.get("family", ""),
                "sigma": r.get("sigma", ""),
            }
        )
    extra = pd.DataFrame(search_rows)
    combined = pd.concat([base, extra], ignore_index=True)
    combined = combined.drop_duplicates(subset=["sample_id"], keep="last")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    return combined


def append_sim_results(
    base_csv: Path,
    new_rows: pd.DataFrame,
    *,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Merge new MEEP rows into a corpus CSV (dedupe by sample_id, keep newest)."""
    out = out_path or base_csv
    if base_csv.exists():
        base = pd.read_csv(base_csv)
    else:
        base = pd.DataFrame()
    combined = pd.concat([base, new_rows], ignore_index=True)
    if "sample_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["sample_id"], keep="last")
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    return combined
