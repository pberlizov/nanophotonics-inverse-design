"""Types for multi-solver cross-check."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class SolverSpec:
    name: str
    family: str  # meep | tidy3d | bpm | manual
    recipe_version: str = "phase0_v1"
    resolution: int = 25
    notes: str = ""


@dataclass
class CrosscheckResult:
    sample_id: str
    mask_path: str
    solver: str
    status: str
    split_ratio_upper: float
    insertion_loss_db: float
    flux_in: float
    flux_out_upper: float
    flux_out_lower: float
    reference_split: float | None = None
    abs_err_vs_reference: float | None = None
    error: str = ""
    runtime_note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
