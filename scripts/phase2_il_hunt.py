#!/usr/bin/env python3
"""
Phase 2 IL hunt:
  A) weight_il=0.75 multi-objective
  C) Perlin + far-sigma seeds (no champion centers)
  B) stage-2 IL refine on in-spec stage-1 seeds
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
from nano_inv.latent import pad_latent_to_standard, sample_latent_perturbation  # noqa: E402
from nano_inv.manifold import sample_latent_perlin  # noqa: E402
from nano_inv.meep_latent import suggest_latent_for_meep  # noqa: E402
from nano_inv.meep_sim import MeepRecipe, require_meep, simulate_mask  # noqa: E402
from nano_inv.search_objectives import SearchConfig, meep_search_loss  # noqa: E402


def decode_latent(latent_path: Path, mask_path: Path) -> np.ndarray:
    subprocess.run(
        [str(VENV_PY), str(REPO / "scripts/decode_one.py"), "--latent", str(latent_path), "--mask", str(mask_path)],
        cwd=REPO,
        check=True,
    )
    return np.load(mask_path)


def stage1_objective(
    trial: optuna.Trial,
    *,
    ref: np.ndarray,
    recipe: MeepRecipe,
    search_cfg: SearchConfig,
    target: float,
    tol: float,
    sigma_range: tuple[float, float],
    out_dir: Path,
    rng: np.random.Generator,
) -> float:
    family = trial.suggest_categorical("family", ["perlin", "sigma"])
    if family == "perlin":
        z = sample_latent_perlin(
            scale=trial.suggest_float("perlin_scale", 2.5, 5.0),
            offset=(
                trial.suggest_float("perlin_ox", -2.0, 2.0),
                trial.suggest_float("perlin_oy", -2.0, 2.0),
            ),
        )
        trial.set_user_attr("latent_mode", "perlin")
    else:
        sigma = trial.suggest_float("sigma", sigma_range[0], sigma_range[1], log=True)
        z = sample_latent_perturbation(ref, rng, sigma=sigma)
        trial.set_user_attr("latent_mode", "sigma")
        trial.set_user_attr("sigma", sigma)

    z_std = pad_latent_to_standard(z)
    sid = f"il2_s1_{trial.number:04d}"
    latent_path = out_dir / "stage1/latents" / f"{sid}_latent.npy"
    mask_path = out_dir / "stage1/masks" / f"{sid}_mask.npy"
    latent_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(latent_path, z_std)
    mask = decode_latent(latent_path, mask_path)
    if not check_mask_heuristic(mask).passed:
        return 1.0 + search_cfg.drc_penalty

    res = simulate_mask(mask, recipe, verbose=False)
    if res.status != "ok" or not np.isfinite(res.split_ratio_upper):
        return 3.0

    loss = meep_search_loss(res.split_ratio_upper, res.insertion_loss_db, search_cfg)
    split_err = abs(res.split_ratio_upper - target)
    trial.set_user_attr("sample_id", sid)
    trial.set_user_attr("R_up", res.split_ratio_upper)
    trial.set_user_attr("split_err", split_err)
    trial.set_user_attr("IL_db", res.insertion_loss_db)
    trial.set_user_attr("in_spec", split_err <= tol)
    trial.set_user_attr("latent_path", str(latent_path.relative_to(REPO)))
    return float(loss)


def stage2_refine(
    seed_latent: Path,
    seed_id: str,
    *,
    recipe: MeepRecipe,
    target: float,
    tol: float,
    split_penalty: float,
    n_trials: int,
    out_dir: Path,
) -> list[dict]:
    ref = np.load(seed_latent).astype(np.float32)
    rows: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        rng = np.random.default_rng(1000 + trial.number)
        z, _ = suggest_latent_for_meep(
            trial, ref, rng, mode="residual", residual_dims=16, residual_bound=0.04
        )
        z_std = pad_latent_to_standard(z)
        sid = f"il2_s2_{seed_id}_{trial.number:03d}"
        latent_path = out_dir / "stage2/latents" / f"{sid}_latent.npy"
        mask_path = out_dir / "stage2/masks" / f"{sid}_mask.npy"
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(latent_path, z_std)
        mask = decode_latent(latent_path, mask_path)
        if not check_mask_heuristic(mask).passed:
            return 2.0

        res = simulate_mask(mask, recipe, verbose=False)
        if res.status != "ok" or not np.isfinite(res.insertion_loss_db):
            return 3.0
        split_err = abs(res.split_ratio_upper - target)
        loss = float(res.insertion_loss_db + split_penalty * max(0.0, split_err - tol))
        trial.set_user_attr("sample_id", sid)
        trial.set_user_attr("IL_db", res.insertion_loss_db)
        trial.set_user_attr("split_err", split_err)
        trial.set_user_attr("in_spec", split_err <= tol)
        return loss

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=99))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    for t in study.trials:
        if not t.user_attrs.get("sample_id"):
            continue
        rows.append(
            {
                "seed_id": seed_id,
                "sample_id": t.user_attrs["sample_id"],
                "stage": 2,
                "loss": t.value,
                "IL_db": t.user_attrs.get("IL_db"),
                "split_err": t.user_attrs.get("split_err"),
                "in_spec": t.user_attrs.get("in_spec"),
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO / "configs/phase2_il_hunt.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    hunt = cfg.get("il_hunt") or {}
    search = cfg.get("search") or {}
    targets = cfg.get("targets") or {}
    out_dir = REPO / hunt.get("output_dir", "data/phase1/phase2_il_hunt")
    out_dir.mkdir(parents=True, exist_ok=True)

    target = float(targets.get("split_ratio_1550", 0.5))
    tol = float(targets.get("split_ratio_tolerance", 0.05))
    recipe = MeepRecipe.for_version(
        (cfg.get("meep") or {}).get("recipe_version", "phase0_v1"),
        dict(cfg.get("meep") or {}),
    )
    search_cfg = SearchConfig(
        target_split_ratio=target,
        tolerance=tol,
        objective="multi",
        weight_split=float(search.get("weight_split", 1.0)),
        weight_il=float(search.get("weight_il", 0.75)),
        max_insertion_loss_db=float(targets.get("max_insertion_loss_db", 12.0)),
        drc_penalty=float(search.get("drc_penalty", 1.0)),
    )
    sigma_range = tuple(hunt.get("sigma_range", [0.02, 0.12]))

    require_meep()
    ref = np.load(REPO / "data/phase0/latents/ref_published_latent.npy").astype(np.float32)

    print(f"==> Stage 1: {hunt.get('stage1_trials', 100)} Perlin/sigma trials (weight_il={search_cfg.weight_il})")
    study1 = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))

    def obj1(trial: optuna.Trial) -> float:
        return stage1_objective(
            trial,
            ref=ref,
            recipe=recipe,
            search_cfg=search_cfg,
            target=target,
            tol=tol,
            sigma_range=sigma_range,
            out_dir=out_dir,
            rng=np.random.default_rng(42 + trial.number),
        )

    study1.optimize(obj1, n_trials=int(hunt.get("stage1_trials", 100)), show_progress_bar=True)

    s1_rows = []
    for t in study1.trials:
        if not t.user_attrs.get("sample_id"):
            continue
        s1_rows.append(
            {
                "stage": 1,
                "sample_id": t.user_attrs["sample_id"],
                "loss": t.value,
                "IL_db": t.user_attrs.get("IL_db"),
                "split_err": t.user_attrs.get("split_err"),
                "in_spec": t.user_attrs.get("in_spec"),
                "latent_mode": t.user_attrs.get("latent_mode"),
                "latent_path": t.user_attrs.get("latent_path"),
            }
        )
    s1_df = pd.DataFrame(s1_rows)
    s1_df.to_csv(out_dir / "stage1_trials.csv", index=False)

    seeds = s1_df[s1_df["in_spec"] == True].sort_values("IL_db").head(int(hunt.get("stage2_top_seeds", 5)))
    if len(seeds) == 0:
        seeds = s1_df.sort_values("loss").head(int(hunt.get("stage2_top_seeds", 5)))
        print("  (no in-spec stage-1; using best loss seeds for stage 2)")

    s2_rows: list[dict] = []
    for _, row in seeds.iterrows():
        lp = REPO / row["latent_path"]
        if not lp.exists():
            continue
        print(f"==> Stage 2 IL refine from {row['sample_id']}")
        s2_rows.extend(
            stage2_refine(
                lp,
                row["sample_id"],
                recipe=recipe,
                target=target,
                tol=tol,
                split_penalty=float(hunt.get("stage2_split_penalty", 10.0)),
                n_trials=int(hunt.get("stage2_trials_per_seed", 30)),
                out_dir=out_dir,
            )
        )

    s2_df = pd.DataFrame(s2_rows)
    s2_df.to_csv(out_dir / "stage2_trials.csv", index=False)
    all_df = pd.concat([s1_df, s2_df], ignore_index=True)
    all_df.to_csv(out_dir / "phase2_il_trials.csv", index=False)

    best_il = float(s2_df["IL_db"].min()) if len(s2_df) else float(s1_df["IL_db"].min())
    summary = {
        "stage1_trials": len(s1_df),
        "stage1_in_spec": int(s1_df["in_spec"].fillna(False).sum()),
        "stage2_trials": len(s2_df),
        "weight_il": search_cfg.weight_il,
        "best_IL_db": best_il,
        "output_dir": str(out_dir.relative_to(REPO)),
    }
    (out_dir / "phase2_il_summary.json").write_text(json.dumps(summary, indent=2))
    (REPO / "data/phase1/release/phase2_il_hunt.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
