"""Candidate pool helpers — diverse shortlists for surrogate_rank."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nano_inv.surrogate import mask_to_features, pool_mask


def pooled_hamming(a: np.ndarray, b: np.ndarray, pool: int = 6) -> float:
    """Fraction of disagreeing pixels after max-pool (approximate mask distance)."""
    pa = (pool_mask(a, pool) > 0.5).astype(np.uint8)
    pb = (pool_mask(b, pool) > 0.5).astype(np.uint8)
    return float(np.mean(pa != pb))


def diverse_top_k(
    df: pd.DataFrame,
    k: int,
    *,
    repo_root: Path,
    score_col: str = "surrogate_score",
    mask_col: str = "mask_path",
    latent_col: str = "latent_path",
    min_hamming: float = 0.02,
    pool: int = 6,
) -> pd.DataFrame:
    """
    Greedy diverse selection: walk sorted-by-score rows, accept if not too similar
    to an already picked mask (pooled Hamming >= min_hamming).
    """
    if len(df) == 0:
        return df
    ordered = df.sort_values(score_col).reset_index(drop=True)
    picked: list[int] = []
    masks: list[np.ndarray] = []

    for idx, row in ordered.iterrows():
        if len(picked) >= k:
            break
        mp = str(row.get(mask_col, "") or "").strip()
        if mp:
            path = repo_root / mp
            if path.is_file():
                m = np.load(path)
            else:
                continue
        else:
            lp = str(row.get(latent_col, "") or "").strip()
            if not lp:
                continue
            from nano_inv.manifold import EBeamManifold

            m = EBeamManifold.load().decode_numpy(np.load(repo_root / lp))

        too_close = False
        for prev in masks:
            if pooled_hamming(m, prev, pool=pool) < min_hamming:
                too_close = True
                break
        if too_close:
            continue
        picked.append(int(idx))
        masks.append(m)

    if not picked:
        return ordered.head(k)
    out = ordered.iloc[picked].reset_index(drop=True)
    if len(out) < k:
        rest = ordered.drop(index=picked).head(k - len(out))
        out = pd.concat([out, rest], ignore_index=True)
    return out
