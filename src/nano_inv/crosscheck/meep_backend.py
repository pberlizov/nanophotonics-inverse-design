"""MEEP backend for cross-check."""

from __future__ import annotations

import numpy as np

from nano_inv.crosscheck.types import CrosscheckResult, SolverSpec
from nano_inv.meep_sim import MeepRecipe, simulate_mask


def run_meep(
    sample_id: str,
    mask_path: str,
    mask: np.ndarray,
    spec: SolverSpec,
    *,
    reference_split: float | None = None,
    verbose: bool = False,
) -> CrosscheckResult:
    recipe = MeepRecipe.for_version(
        spec.recipe_version,
        {"resolution": spec.resolution},
    )
    res = simulate_mask(mask, recipe, verbose=verbose, sample_key=sample_id)
    err = None
    if reference_split is not None and np.isfinite(res.split_ratio_upper):
        err = float(abs(res.split_ratio_upper - reference_split))
    return CrosscheckResult(
        sample_id=sample_id,
        mask_path=mask_path,
        solver=spec.name,
        status=res.status,
        split_ratio_upper=float(res.split_ratio_upper),
        insertion_loss_db=float(res.insertion_loss_db),
        flux_in=float(res.flux_in),
        flux_out_upper=float(res.flux_out_upper),
        flux_out_lower=float(res.flux_out_lower),
        reference_split=reference_split,
        abs_err_vs_reference=err,
        error=res.error,
        runtime_note=f"recipe={spec.recipe_version} res={spec.resolution}",
    )
