#!/usr/bin/env python3
"""Optuna search for split + morphology robustness (nominal + erode/dilate stress)."""

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
from nano_inv.fab_stress import apply_morph_stress  # noqa: E402
from nano_inv.latent import pad_latent_to_standard  # noqa: E402
from nano_inv.meep_latent import suggest_latent_for_meep  # noqa: E402
from nano_inv.meep_sim import MeepRecipe, require_meep, simulate_mask  # noqa: E402


def decode_latent_to_mask(latent_path: Path, mask_path: Path) -> np.ndarray:
    subprocess.run(
        [str(VENV_PY), str(REPO / "scripts/decode_one.py"), "--latent", str(latent_path), "--mask", str(mask_path)],
        cwd=REPO,
        check=True,
    )
    return np.load(mask_path)


def morph_robust_loss(
    mask: np.ndarray,
    recipe: MeepRecipe,
    *,
    target: float,
    stress_nm_levels: list[int],
    upscale: int,
    max_delta: float,
    tol: float,
    weight_max_delta: float,
    weight_worst: float,
) -> tuple[float, dict]:
    """Worst |R_up-target| and max |ΔR_up| over nominal + erode/dilate at each stress level."""
    details: dict = {"stress_nm_levels": stress_nm_levels}
    worst = 0.0
    max_d = 0.0
    r_nom: float | None = None

    for stress_nm in stress_nm_levels:
        for op, label in (("none", "nominal"), ("erode", "erode"), ("dilate", "dilate")):
            if op == "none" and stress_nm != stress_nm_levels[0]:
                continue
            if op == "none":
                stressed = mask.astype(float)
                key = "R_up_nominal"
            else:
                stressed, _, _ = apply_morph_stress(mask, float(stress_nm), op, upscale=upscale)
                key = f"R_up_{label}_{stress_nm}nm"

            res = simulate_mask(stressed, recipe, verbose=False)
            if res.status != "ok" or not np.isfinite(res.split_ratio_upper):
                return 2.0, {"status": "sim_fail"}

            err = abs(res.split_ratio_upper - target)
            worst = max(worst, err)
            details[key] = res.split_ratio_upper

            if op == "none":
                r_nom = res.split_ratio_upper
            elif r_nom is not None:
                delta = abs(res.split_ratio_upper - r_nom)
                max_d = max(max_d, delta)
                details[f"delta_{label}_{stress_nm}nm"] = delta

    details["worst_split_error"] = worst
    details["max_delta_R_up"] = max_d
    details["morph_pass"] = worst <= tol and max_d <= max_delta
    loss = float(weight_max_delta * max_d + weight_worst * worst)
    return loss, details


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO / "configs/morph_robust_hunt.yaml")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--trials-per-center", type=int, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    hunt = cfg.get("hunt") or {}
    morph = cfg.get("morph") or {}
    search = cfg.get("search") or {}
    targets = cfg.get("targets") or {}
    out_dir = Path(args.output_dir or hunt.get("output_dir", "data/phase1/morph_robust_hunt"))
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    target = float(targets.get("split_ratio_1550", 0.5))
    tol = float(targets.get("split_ratio_tolerance", 0.05))
    meep_cfg = dict(cfg.get("meep") or {})
    recipe = MeepRecipe.for_version(meep_cfg.get("recipe_version", "phase0_v1"), meep_cfg)
    stress_levels = [int(x) for x in morph.get("stress_nm_levels", [10])]
    upscale = int(morph.get("upscale", 5))
    max_delta = float(morph.get("max_delta_R_up", 0.05))
    w_delta = float(search.get("weight_max_delta", 1.0))
    w_worst = float(search.get("weight_worst", 2.0))
    trials_each = args.trials_per_center or int(hunt.get("trials_per_center", 30))

    require_meep()
    all_rows: list[dict] = []

    for center_path in [REPO / p for p in hunt.get("center_latents", [])]:
        if not center_path.exists():
            print(f"skip missing center {center_path}")
            continue
        center_id = center_path.stem.replace("_latent", "")
        ref = np.load(center_path).astype(np.float32)
        print(
            f"==> morph-robust refine from {center_id} ({trials_each} trials, "
            f"stress={stress_levels} nm, loss={w_delta}*Δ+{w_worst}*worst)"
        )

        def objective(trial: optuna.Trial) -> float:
            rng = np.random.default_rng(42 + trial.number)
            z, _ = suggest_latent_for_meep(
                trial,
                ref,
                rng,
                mode="residual",
                residual_dims=int(hunt.get("residual_dims", 16)),
                residual_bound=float(hunt.get("residual_bound", 0.035)),
            )
            z_std = pad_latent_to_standard(z)
            sid = f"morph_{center_id}_{trial.number:03d}"
            latent_path = out_dir / "candidates/latents" / f"{sid}_latent.npy"
            mask_path = out_dir / "candidates/masks" / f"{sid}_mask.npy"
            latent_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(latent_path, z_std)
            mask = decode_latent_to_mask(latent_path, mask_path)
            if not check_mask_heuristic(mask).passed:
                return 1.0 + float(search.get("drc_penalty", 1.0))

            loss, det = morph_robust_loss(
                mask,
                recipe,
                target=target,
                stress_nm_levels=stress_levels,
                upscale=upscale,
                max_delta=max_delta,
                tol=tol,
                weight_max_delta=w_delta,
                weight_worst=w_worst,
            )
            if det.get("status") == "sim_fail":
                return 3.0
            trial.set_user_attr("sample_id", sid)
            trial.set_user_attr("morph_pass", det.get("morph_pass"))
            trial.set_user_attr("worst_split_error", det.get("worst_split_error"))
            trial.set_user_attr("max_delta_R_up", det.get("max_delta_R_up"))
            trial.set_user_attr("R_up_nominal", det.get("R_up_nominal"))
            return loss

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=trials_each, show_progress_bar=True)

        for t in study.trials:
            if not t.user_attrs.get("sample_id"):
                continue
            all_rows.append(
                {
                    "center_id": center_id,
                    "sample_id": t.user_attrs["sample_id"],
                    "loss": t.value,
                    "morph_pass": t.user_attrs.get("morph_pass"),
                    "worst_split_error": t.user_attrs.get("worst_split_error"),
                    "max_delta_R_up": t.user_attrs.get("max_delta_R_up"),
                    "R_up_nominal": t.user_attrs.get("R_up_nominal"),
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
    if len(df):
        df = df.sort_values(["morph_pass", "worst_split_error", "max_delta_R_up"], ascending=[False, True, True])
    df.to_csv(out_dir / "morph_robust_trials.csv", index=False)
    top = df.dropna(subset=["worst_split_error"]).head(10)
    top.to_csv(out_dir / "morph_robust_top.csv", index=False)

    n_pass = int(df["morph_pass"].fillna(False).sum()) if len(df) else 0
    summary = {
        "trials_per_center": trials_each,
        "stress_nm_levels": stress_levels,
        "weight_max_delta": w_delta,
        "weight_worst": w_worst,
        "residual_bound": float(hunt.get("residual_bound", 0.035)),
        "n_pass": n_pass,
        "n_trials": len(df),
        "best_worst_split_error": float(df["worst_split_error"].min()) if len(df) else None,
        "best_max_delta_R_up": float(df["max_delta_R_up"].min()) if len(df) else None,
        "output_dir": str(out_dir.relative_to(REPO)),
    }
    (out_dir / "morph_robust_summary.json").write_text(json.dumps(summary, indent=2))
    release = REPO / "data/phase1/release/morph_robust_hunt.json"
    release.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
