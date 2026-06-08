"""Merge MEEP search / champion labels into the training corpus CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SIM_COLUMNS = [
    "sample_id",
    "source",
    "mask_path",
    "recipe_version",
    "resolution",
    "status",
    "flux_in",
    "flux_out_upper",
    "flux_out_lower",
    "split_ratio_upper",
    "insertion_loss_db",
    "error",
    "sigma",
]


def _rel_mask_path(repo_root: Path, path: str | Path) -> str:
    p = Path(path)
    if p.is_absolute():
        try:
            return str(p.relative_to(repo_root))
        except ValueError:
            return str(p)
    return str(p).replace("\\", "/")


def row_from_search_trial(
    repo_root: Path,
    *,
    sample_id: str,
    mask_path: str,
    split_ratio_upper: float,
    source: str = "meep_search",
    recipe_version: str = "phase0_v1",
    resolution: int = 25,
    insertion_loss_db: float | None = None,
    status: str = "ok",
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "source": source,
        "mask_path": _rel_mask_path(repo_root, mask_path),
        "recipe_version": recipe_version,
        "resolution": resolution,
        "status": status,
        "flux_in": np.nan,
        "flux_out_upper": np.nan,
        "flux_out_lower": np.nan,
        "split_ratio_upper": float(split_ratio_upper),
        "insertion_loss_db": insertion_loss_db if insertion_loss_db is not None else np.nan,
        "error": "",
    }


def rows_from_top_candidates_csv(
    repo_root: Path,
    csv_path: Path,
    *,
    source: str = "meep_search",
    recipe_version: str = "phase0_v1",
    resolution: int = 25,
) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    rows: list[dict[str, Any]] = []
    split_col = "split_ratio_upper" if "split_ratio_upper" in df.columns else "meep_split_ratio_upper"
    for _, r in df.iterrows():
        split = r.get(split_col)
        if pd.isna(split):
            continue
        mp = r.get("mask_path", "")
        rows.append(
            row_from_search_trial(
                repo_root,
                sample_id=str(r["sample_id"]),
                mask_path=str(mp),
                split_ratio_upper=float(split),
                source=source,
                recipe_version=str(r.get("recipe_version", recipe_version)),
                resolution=int(r.get("resolution", resolution)),
            )
        )
    return rows


def rows_from_meep_search_trials_csv(
    repo_root: Path,
    csv_path: Path,
    *,
    source: str = "meep_search",
    recipe_version: str = "phase0_v1",
    resolution: int = 25,
    only_ok: bool = True,
) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    split_col = "meep_split_ratio_upper" if "meep_split_ratio_upper" in df.columns else "split_ratio_upper"
    status_col = "status" if "status" in df.columns else None
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        if only_ok and status_col and str(r.get(status_col, "ok")) != "ok":
            continue
        split = r.get(split_col)
        if pd.isna(split):
            continue
        sid = str(r.get("sample_id", ""))
        if not sid:
            continue
        # infer mask path from search dir layout
        parent = csv_path.parent
        mp = parent / "candidates" / "masks" / f"{sid}_mask.npy"
        if not mp.exists():
            mp = repo_root / "data" / "phase0" / "masks" / f"{sid}_mask.npy"
        if not mp.exists():
            continue
        rows.append(
            row_from_search_trial(
                repo_root,
                sample_id=sid,
                mask_path=str(mp.relative_to(repo_root) if mp.is_relative_to(repo_root) else mp),
                split_ratio_upper=float(split),
                source=source,
                recipe_version=recipe_version,
                resolution=resolution,
            )
        )
    return rows


def default_champion_sources() -> list[dict[str, Any]]:
    return [
        {
            "kind": "top_candidates",
            "path": "data/phase1/meep_search_local/top_candidates.csv",
            "source": "meep_search_local",
        },
        {
            "kind": "trials",
            "path": "data/phase1/meep_search_local/meep_search_local_trials.csv",
            "source": "meep_search_local",
        },
        {
            "kind": "top_candidates",
            "path": "data/phase1/meep_search_deep/top_candidates.csv",
            "source": "meep_search_deep",
        },
        {
            "kind": "trials",
            "path": "data/phase1/meep_search_deep/meep_search_trials.csv",
            "source": "meep_search_deep",
        },
        {
            "kind": "top_candidates",
            "path": "data/phase0/meep_search_100/top_candidates.csv",
            "source": "meep_search_phase0",
        },
    ]


def _parse_budget_from_policy_dir(name: str) -> int | None:
    if "_B" not in name:
        return None
    try:
        return int(name.rsplit("_B", 1)[-1])
    except ValueError:
        return None


def _replicate_seed(base_seed: int, run_name: str) -> int:
    try:
        rid = int(run_name.replace("run_", ""))
    except ValueError:
        rid = 1
    return int(base_seed) + rid * 1000


def _reconstruct_latent(
    repo_root: Path,
    *,
    sample_id: str,
    sigma: float | None,
    policy_dir: Path,
    replicate_seed: int,
    ref: np.ndarray,
) -> np.ndarray | None:
    """Rebuild latent when mask files were pruned (deterministic from run protocol)."""
    from nano_inv.latent import pad_latent_to_standard, sample_latent_perturbation

    budget = _parse_budget_from_policy_dir(policy_dir.name)
    if budget is None:
        return None

    pool_csv = policy_dir.parent / "candidates_pool.csv"
    if pool_csv.exists():
        pool = pd.read_csv(pool_csv)
        hit = pool.loc[pool["sample_id"].astype(str) == sample_id]
        if len(hit):
            lp = hit.iloc[0].get("latent_path", "")
            if lp and (repo_root / str(lp)).is_file():
                return np.load(repo_root / str(lp)).astype(np.float32)

    hier = list(policy_dir.glob("candidates_hier_*.csv"))
    for hp in hier:
        hdf = pd.read_csv(hp)
        hit = hdf.loc[hdf["sample_id"].astype(str) == sample_id]
        if len(hit):
            lp = hit.iloc[0].get("latent_path", "")
            if lp and (repo_root / str(lp)).is_file():
                return np.load(repo_root / str(lp)).astype(np.float32)

    if sigma is None or not np.isfinite(sigma):
        return None

    if sample_id.startswith("sig_"):
        parts = sample_id.split("_")
        if len(parts) >= 3:
            trial_num = int(parts[2])
            rng = np.random.default_rng(replicate_seed + budget + trial_num)
            return pad_latent_to_standard(sample_latent_perturbation(ref, rng, sigma=float(sigma)))

    if sample_id.startswith("rnd_"):
        parts = sample_id.split("_")
        if len(parts) >= 3:
            idx = int(parts[2])
            rng = np.random.default_rng(replicate_seed)
            for _ in range(idx + 1):
                s = float(rng.uniform(0.008, 0.04))
            return pad_latent_to_standard(sample_latent_perturbation(ref, rng, sigma=float(sigma)))

    if sample_id.startswith("cand_"):
        if pool_csv.exists():
            pass  # already tried
    return None


def materialize_mask(
    repo_root: Path,
    latent: np.ndarray,
    sample_id: str,
    *,
    cache_dir: Path,
) -> tuple[str, str]:
    """Decode latent → mask under cache_dir; return (mask_path, latent_path) repo-relative."""
    from nano_inv.manifold import EBeamManifold

    cache_dir.mkdir(parents=True, exist_ok=True)
    latent_dir = cache_dir.parent / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    lp = latent_dir / f"{sample_id}_latent.npy"
    np.save(lp, latent)
    mp = cache_dir / f"{sample_id}_mask.npy"
    if not mp.exists():
        mask = EBeamManifold.load().decode_numpy(latent)
        np.save(mp, mask)
    try:
        return str(mp.relative_to(repo_root)), str(lp.relative_to(repo_root))
    except ValueError:
        return str(mp), str(lp)


def rows_from_sim_budget_replicates(
    repo_root: Path,
    replicates_root: Path,
    *,
    source: str = "sim_budget",
    recipe_version: str = "phase0_v1",
    resolution: int = 25,
    base_seed: int = 2026,
    materialize_masks: bool = True,
    mask_cache_dir: str = "data/phase1/corpus_materialized/masks",
) -> list[dict[str, Any]]:
    """Collect MEEP rows from run_XX/*_B*/meep_results.csv across replicate studies."""
    ref_path = repo_root / "data/phase0/latents/ref_published_latent.npy"
    if not ref_path.exists():
        return []
    ref = np.load(ref_path).astype(np.float32)
    cache = repo_root / mask_cache_dir
    rows: list[dict[str, Any]] = []

    for run_dir in sorted(replicates_root.glob("run_*")):
        if not run_dir.is_dir():
            continue
        rep_seed = _replicate_seed(base_seed, run_dir.name)
        for policy_dir in sorted(run_dir.iterdir()):
            if not policy_dir.is_dir() or "_B" not in policy_dir.name:
                continue
            csv_path = policy_dir / "meep_results.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            for _, r in df.iterrows():
                split = r.get("split_ratio_upper")
                if pd.isna(split) or not np.isfinite(float(split)):
                    continue
                sid = str(r["sample_id"])
                sigma = r.get("sigma")
                sigma_f = float(sigma) if sigma is not None and pd.notna(sigma) else None

                mp_rel = ""
                for sub in ("candidates/masks", "phase_sigma/candidates/masks", "phase_rank/candidates/masks"):
                    cand = policy_dir / sub / f"{sid}_mask.npy"
                    if cand.exists():
                        mp_rel = _rel_mask_path(repo_root, cand)
                        break

                if not mp_rel and materialize_masks:
                    z = _reconstruct_latent(
                        repo_root,
                        sample_id=sid,
                        sigma=sigma_f,
                        policy_dir=policy_dir,
                        replicate_seed=rep_seed,
                        ref=ref,
                    )
                    if z is not None:
                        uid = f"{sid}_rep{run_dir.name.replace('run_', '')}"
                        mp_rel, _ = materialize_mask(repo_root, z, uid, cache_dir=cache)

                if not mp_rel:
                    continue

                rid = run_dir.name.replace("run_", "")
                uid = f"{sid}_rep{rid}"
                row = row_from_search_trial(
                    repo_root,
                    sample_id=uid,
                    mask_path=mp_rel,
                    split_ratio_upper=float(split),
                    source=source,
                    recipe_version=recipe_version,
                    resolution=resolution,
                    insertion_loss_db=float(r["insertion_loss_db"])
                    if pd.notna(r.get("insertion_loss_db"))
                    else None,
                )
                if sigma_f is not None:
                    row["sigma"] = sigma_f
                rows.append(row)
    return rows


def collect_merge_rows(repo_root: Path, sources: list[dict[str, Any]]) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    for spec in sources:
        kind = spec.get("kind", "top_candidates")
        src = spec.get("source", "meep_search")
        if kind == "sim_budget_replicates":
            rep_root = repo_root / spec.get(
                "path", "data/phase1/wedge_a/sim_budget/replicates"
            )
            all_rows.extend(
                rows_from_sim_budget_replicates(
                    repo_root,
                    rep_root,
                    source=spec.get("source", src),
                    base_seed=int(spec.get("base_seed", 2026)),
                    materialize_masks=bool(spec.get("materialize_masks", True)),
                    mask_cache_dir=str(
                        spec.get("mask_cache_dir", "data/phase1/corpus_materialized/masks")
                    ),
                )
            )
            continue
        p = repo_root / spec["path"]
        if not p.exists():
            continue
        if kind == "top_candidates":
            all_rows.extend(rows_from_top_candidates_csv(repo_root, p, source=src))
        elif kind == "trials":
            all_rows.extend(rows_from_meep_search_trials_csv(repo_root, p, source=src))
        elif kind == "sim_csv":
            sub = pd.read_csv(p)
            all_rows.extend(sub.to_dict("records"))
    if not all_rows:
        return pd.DataFrame(columns=SIM_COLUMNS)
    df = pd.DataFrame(all_rows)
    for c in SIM_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df[SIM_COLUMNS].drop_duplicates(subset=["sample_id"], keep="last")


def merge_into_corpus(
    repo_root: Path,
    corpus_path: Path,
    new_rows: pd.DataFrame,
    *,
    backup: bool = True,
) -> pd.DataFrame:
    corpus_path = Path(corpus_path)
    if backup and corpus_path.exists():
        bak = corpus_path.with_suffix(".csv.bak")
        bak.write_text(corpus_path.read_text())
    if corpus_path.exists():
        base = pd.read_csv(corpus_path)
    else:
        base = pd.DataFrame(columns=SIM_COLUMNS)
    combined = pd.concat([base, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["sample_id"], keep="last")
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(corpus_path, index=False)
    return combined


EXTRA_MANIFEST_COLS = ("sigma",)
