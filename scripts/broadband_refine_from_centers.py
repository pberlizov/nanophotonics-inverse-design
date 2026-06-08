#!/usr/bin/env python3
"""Broadband Optuna refinement around promoted champion latents (residual mode)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
VENV_PY = REPO / ".venv/bin/python"
sys.path.insert(0, str(REPO / "src"))

from nano_inv.drc_heuristic import check_mask_heuristic  # noqa: E402
from nano_inv.latent import pad_latent_to_standard  # noqa: E402
from nano_inv.meep_latent import suggest_latent_for_meep  # noqa: E402
from nano_inv.meep_sim import MeepRecipe, require_meep, simulate_mask_broadband  # noqa: E402
from nano_inv.search_objectives import broadband_split_loss  # noqa: E402


def decode_latent_to_mask(latent_path: Path, mask_path: Path) -> np.ndarray:
    subprocess.run(
        [
            str(VENV_PY),
            str(REPO / "scripts/decode_one.py"),
            "--latent",
            str(latent_path),
            "--mask",
            str(mask_path),
        ],
        cwd=REPO,
        check=True,
    )
    return np.load(mask_path)


def c_band_wavelengths(cfg: dict) -> list[float]:
    bb = cfg.get("broadband") or {}
    wls = np.arange(float(bb["wl_start"]), float(bb["wl_stop"]) + 1e-9, float(bb["wl_step"]))
    return [round(float(w), 3) for w in wls]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO / "configs/broadband_hunt.yaml")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--trials-per-center", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    hunt = cfg.get("hunt") or {}
    out_dir = Path(args.output_dir or hunt.get("output_dir", "data/phase1/broadband_hunt/refine"))
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = cfg.get("targets") or {}
    target = float(targets.get("split_ratio_1550", 0.5))
    tol = float(targets.get("split_ratio_tolerance", 0.05))
    search = cfg.get("search") or {}
    flatness_w = float(search.get("flatness_weight", 0.0))
    weight_il = float(search.get("weight_il", 0.10))
    max_il = float(targets.get("max_insertion_loss_db", 12.0))
    bb_gate = float((cfg.get("broadband") or {}).get("max_worst_split_error", tol))

    meep_cfg = dict(cfg.get("meep") or {})
    recipe = MeepRecipe.for_version(meep_cfg.get("recipe_version", "phase0_v1"), meep_cfg)
    wavelengths = c_band_wavelengths(cfg)

    centers = [REPO / p for p in hunt.get("center_latents", [])]
    trials_each = args.trials_per_center or int(hunt.get("refine_trials_per_center", 25))
    residual_dims = int(hunt.get("residual_dims", 16))
    residual_bound = float(hunt.get("residual_bound", 0.04))
    latent_mode = hunt.get("latent_mode", "residual")

    require_meep()
    all_rows: list[dict] = []
    global_trial = 0

    for center_path in centers:
        if not center_path.exists():
            print(f"skip missing center {center_path}")
            continue
        center_id = center_path.stem.replace("_latent", "")
        ref = np.load(center_path).astype(np.float32)
        print(f"==> refine from {center_id} ({trials_each} trials)")

        def objective(trial: optuna.Trial) -> float:
            nonlocal global_trial
            rng = np.random.default_rng(42 + global_trial)
            global_trial += 1
            z, _meta = suggest_latent_for_meep(
                trial,
                ref,
                rng,
                mode=latent_mode,
                residual_dims=residual_dims,
                residual_bound=residual_bound,
            )
            z_std = pad_latent_to_standard(z)
            sid = f"bb_{center_id}_{trial.number:03d}"
            latent_path = out_dir / "candidates" / "latents" / f"{sid}_latent.npy"
            mask_path = out_dir / "candidates" / "masks" / f"{sid}_mask.npy"
            latent_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(latent_path, z_std)
            mask = decode_latent_to_mask(latent_path, mask_path)
            if not check_mask_heuristic(mask).passed:
                return 1.0 + float(search.get("drc_penalty", 1.0))

            bb = simulate_mask_broadband(mask, recipe, wavelengths, target_split=target, verbose=False)
            if bb.status != "ok":
                return 2.0
            loss = broadband_split_loss(
                bb.splits_by_wavelength, target, flatness_weight=flatness_w
            )
            loss += weight_il * max(0.0, bb.mean_insertion_loss_db - max_il) / max(max_il, 1e-6)
            trial.set_user_attr("center_id", center_id)
            trial.set_user_attr("sample_id", sid)
            trial.set_user_attr("worst_split_error", bb.worst_split_error)
            trial.set_user_attr("broadband_pass", bb.worst_split_error <= bb_gate)
            trial.set_user_attr("splits_by_wavelength", {str(k): v for k, v in bb.splits_by_wavelength.items()})
            trial.set_user_attr("meep_split_ratio_upper", bb.splits_by_wavelength.get(1.55, float("nan")))
            return float(loss)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=f"bb_refine_{center_id}",
        )
        study.optimize(objective, n_trials=trials_each, show_progress_bar=True)

        for t in study.trials:
            if t.user_attrs.get("sample_id") is None:
                continue
            all_rows.append(
                {
                    "center_id": t.user_attrs.get("center_id"),
                    "sample_id": t.user_attrs["sample_id"],
                    "loss": t.value,
                    "worst_split_error": t.user_attrs.get("worst_split_error"),
                    "broadband_pass": t.user_attrs.get("broadband_pass"),
                    "meep_split_ratio_upper": t.user_attrs.get("meep_split_ratio_upper"),
                    "latent_path": str(
                        (out_dir / "candidates/latents" / f"{t.user_attrs['sample_id']}_latent.npy").relative_to(
                            REPO
                        )
                    ),
                    "mask_path": str(
                        (out_dir / "candidates/masks" / f"{t.user_attrs['sample_id']}_mask.npy").relative_to(REPO)
                    ),
                }
            )

    df = pd.DataFrame(all_rows)
    df = df.sort_values("worst_split_error", na_position="last")
    df.to_csv(out_dir / "broadband_refine_trials.csv", index=False)
    top = df.dropna(subset=["worst_split_error"]).head(10)
    top.to_csv(out_dir / "broadband_refine_top.csv", index=False)

    n_pass = int(df["broadband_pass"].fillna(False).sum()) if len(df) else 0
    summary = {
        "stage": "refine_from_centers",
        "n_centers": len(centers),
        "trials_per_center": trials_each,
        "wavelengths_um": wavelengths,
        "flatness_weight": flatness_w,
        "n_trials": len(df),
        "n_broadband_pass": n_pass,
        "best_worst_split_error": float(df["worst_split_error"].min()) if len(df) else None,
        "best_sample_id": str(df.iloc[0]["sample_id"]) if len(df) else None,
        "output_dir": str(out_dir.relative_to(REPO)),
    }
    (out_dir / "broadband_refine_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
