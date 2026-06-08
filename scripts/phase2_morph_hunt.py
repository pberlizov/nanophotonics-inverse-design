#!/usr/bin/env python3
"""
Phase 2 morph hunt:
  B) Perlin from trial 1
  C) Residual refine from dilate-survivor centers
  D) Asymmetric loss: |R_erode-0.5| + |R_dilate-0.5| + w*|R_nom-0.5|
"""

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
from nano_inv.manifold import sample_latent_perlin  # noqa: E402
from nano_inv.meep_latent import suggest_latent_for_meep  # noqa: E402
from nano_inv.meep_sim import MeepRecipe, require_meep, simulate_mask  # noqa: E402


def decode_latent(latent_path: Path, mask_path: Path) -> np.ndarray:
    subprocess.run(
        [str(VENV_PY), str(REPO / "scripts/decode_one.py"), "--latent", str(latent_path), "--mask", str(mask_path)],
        cwd=REPO,
        check=True,
    )
    return np.load(mask_path)


def morph_asymmetric_loss(
    mask: np.ndarray,
    recipe: MeepRecipe,
    *,
    target: float,
    stress_levels: list[int],
    upscale: int,
    tol: float,
    max_delta: float,
    w_nom: float,
    w_asym: float,
) -> tuple[float, dict]:
    details: dict = {"stress_nm_levels": stress_levels}
    r_nom: float | None = None
    asym_sum = 0.0
    worst = 0.0
    max_d = 0.0

    for stress_nm in stress_levels:
        for op, label in (("none", "nominal"), ("erode", "erode"), ("dilate", "dilate")):
            if op == "none" and stress_nm != stress_levels[0]:
                continue
            if op == "none":
                stressed = mask.astype(float)
                key = "R_up_nominal"
            else:
                stressed, _, _ = apply_morph_stress(mask, float(stress_nm), op, upscale=upscale)
                key = f"R_up_{label}_{stress_nm}nm"

            res = simulate_mask(stressed, recipe, verbose=False)
            if res.status != "ok" or not np.isfinite(res.split_ratio_upper):
                return 3.0, {"status": "sim_fail"}

            err = abs(res.split_ratio_upper - target)
            worst = max(worst, err)
            details[key] = res.split_ratio_upper

            if op == "none":
                r_nom = res.split_ratio_upper
            elif op == "erode":
                asym_sum += err
            elif op == "dilate":
                asym_sum += err
                if r_nom is not None:
                    max_d = max(max_d, abs(res.split_ratio_upper - r_nom))

    nom_err = abs((r_nom or target) - target)
    loss = float(w_asym * asym_sum + w_nom * nom_err)
    details["asym_sum"] = asym_sum
    details["worst_split_error"] = worst
    details["max_delta_R_up"] = max_d
    details["morph_pass"] = worst <= tol and max_d <= max_delta
    return loss, details


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO / "configs/phase2_morph_hunt.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    hunt = cfg.get("hunt") or {}
    morph = cfg.get("morph") or {}
    search = cfg.get("search") or {}
    targets = cfg.get("targets") or {}
    out_dir = REPO / hunt.get("output_dir", "data/phase1/phase2_morph_hunt")
    out_dir.mkdir(parents=True, exist_ok=True)

    target = float(targets.get("split_ratio_1550", 0.5))
    tol = float(targets.get("split_ratio_tolerance", 0.05))
    recipe = MeepRecipe.for_version(
        (cfg.get("meep") or {}).get("recipe_version", "phase0_v1"),
        dict(cfg.get("meep") or {}),
    )
    stress_levels = [int(x) for x in morph.get("stress_nm_levels", [10, 20])]
    upscale = int(morph.get("upscale", 5))
    max_delta = float(morph.get("max_delta_R_up", 0.05))
    w_nom = float(search.get("weight_nominal", 0.5))
    w_asym = float(search.get("weight_asymmetric", 1.0))

    require_meep()
    all_rows: list[dict] = []

    def eval_mask(mask: np.ndarray, sid: str, center_id: str, mode: str) -> float:
        loss, det = morph_asymmetric_loss(
            mask,
            recipe,
            target=target,
            stress_levels=stress_levels,
            upscale=upscale,
            tol=tol,
            max_delta=max_delta,
            w_nom=w_nom,
            w_asym=w_asym,
        )
        if det.get("status") == "sim_fail":
            return 3.0
        all_rows.append(
            {
                "center_id": center_id,
                "sample_id": sid,
                "latent_mode": mode,
                "loss": loss,
                "morph_pass": det.get("morph_pass"),
                "worst_split_error": det.get("worst_split_error"),
                "max_delta_R_up": det.get("max_delta_R_up"),
                "asym_sum": det.get("asym_sum"),
                "R_up_nominal": det.get("R_up_nominal"),
            }
        )
        return float(loss)

    # B: Perlin from scratch
    n_perlin = int(hunt.get("perlin_trials", 80))
    print(f"==> Perlin morph search ({n_perlin} trials)")

    def perlin_obj(trial: optuna.Trial) -> float:
        z = sample_latent_perlin(
            scale=trial.suggest_float("perlin_scale", 2.5, 5.0),
            offset=(trial.suggest_float("ox", -2.0, 2.0), trial.suggest_float("oy", -2.0, 2.0)),
        )
        sid = f"morph2_perlin_{trial.number:04d}"
        lp = out_dir / "candidates/latents" / f"{sid}_latent.npy"
        mp = out_dir / "candidates/masks" / f"{sid}_mask.npy"
        lp.parent.mkdir(parents=True, exist_ok=True)
        np.save(lp, pad_latent_to_standard(z))
        mask = decode_latent(lp, mp)
        if not check_mask_heuristic(mask).passed:
            return 1.0 + float(search.get("drc_penalty", 1.0))
        return eval_mask(mask, sid, "perlin", "perlin")

    perlin_study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=7))
    perlin_study.optimize(perlin_obj, n_trials=n_perlin, show_progress_bar=True)

    # C: Residual from survivor centers
    for center_path in [REPO / p for p in hunt.get("center_latents", [])]:
        if not center_path.exists():
            print(f"skip {center_path}")
            continue
        center_id = center_path.stem.replace("_latent", "")
        ref = np.load(center_path).astype(np.float32)
        n_trials = int(hunt.get("trials_per_center", 100))
        print(f"==> Residual morph from {center_id} ({n_trials} trials)")

        def residual_obj(trial: optuna.Trial) -> float:
            rng = np.random.default_rng(42 + trial.number)
            z, _ = suggest_latent_for_meep(
                trial,
                ref,
                rng,
                mode="residual",
                residual_dims=int(hunt.get("residual_dims", 16)),
                residual_bound=float(hunt.get("residual_bound", 0.05)),
            )
            sid = f"morph2_{center_id}_{trial.number:03d}"
            lp = out_dir / "candidates/latents" / f"{sid}_latent.npy"
            mp = out_dir / "candidates/masks" / f"{sid}_mask.npy"
            lp.parent.mkdir(parents=True, exist_ok=True)
            np.save(lp, pad_latent_to_standard(z))
            mask = decode_latent(lp, mp)
            if not check_mask_heuristic(mask).passed:
                return 1.0 + float(search.get("drc_penalty", 1.0))
            return eval_mask(mask, sid, center_id, "residual")

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(residual_obj, n_trials=n_trials, show_progress_bar=True)

    df = pd.DataFrame(all_rows)
    if len(df):
        df = df.sort_values(["morph_pass", "worst_split_error"], ascending=[False, True])
    df.to_csv(out_dir / "phase2_morph_trials.csv", index=False)

    n_pass = int(df["morph_pass"].fillna(False).sum()) if len(df) else 0
    summary = {
        "n_trials": len(df),
        "n_pass": n_pass,
        "stress_nm_levels": stress_levels,
        "best_worst_split_error": float(df["worst_split_error"].min()) if len(df) else None,
        "output_dir": str(out_dir.relative_to(REPO)),
    }
    (out_dir / "phase2_morph_summary.json").write_text(json.dumps(summary, indent=2))
    (REPO / "data/phase1/release/phase2_morph_hunt.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
