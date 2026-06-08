#!/usr/bin/env python3
"""Phase B: Optuna search with split + IL multi-objective (publication gate)."""

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
from nano_inv.meep_sim import MeepRecipe, require_meep, simulate_mask  # noqa: E402
from nano_inv.search_objectives import SearchConfig, meep_search_loss, pareto_tuple  # noqa: E402


def decode_latent(latent_path: Path, mask_path: Path) -> np.ndarray:
    subprocess.run(
        [str(VENV_PY), str(REPO / "scripts/decode_one.py"), "--latent", str(latent_path), "--mask", str(mask_path)],
        cwd=REPO,
        check=True,
    )
    return np.load(mask_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO / "configs/loss_aware_hunt.yaml")
    p.add_argument("--trials-per-center", type=int, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    hunt = cfg.get("hunt") or {}
    search = cfg.get("search") or {}
    targets = cfg.get("targets") or {}
    out_dir = REPO / hunt.get("output_dir", "data/phase1/loss_aware_hunt")
    out_dir.mkdir(parents=True, exist_ok=True)

    target = float(targets.get("split_ratio_1550", 0.5))
    tol = float(targets.get("split_ratio_tolerance", 0.05))
    meep_cfg = dict(cfg.get("meep") or {})
    recipe = MeepRecipe.for_version(meep_cfg.get("recipe_version", "phase0_v1"), meep_cfg)
    trials_each = args.trials_per_center or int(hunt.get("trials_per_center", 40))

    search_cfg = SearchConfig(
        target_split_ratio=target,
        tolerance=tol,
        objective=str(search.get("objective", "multi")),
        weight_split=float(search.get("weight_split", 1.0)),
        weight_il=float(search.get("weight_il", 0.15)),
        max_insertion_loss_db=float(search.get("max_insertion_loss_db", 12.0)),
        drc_penalty=float(search.get("drc_penalty", 1.0)),
    )

    require_meep()
    rows: list[dict] = []

    for center_path in [REPO / p for p in hunt.get("center_latents", [])]:
        if not center_path.exists():
            print(f"skip missing center {center_path}")
            continue
        center_id = center_path.stem.replace("_latent", "")
        ref = np.load(center_path).astype(np.float32)
        print(f"==> loss-aware refine from {center_id} ({trials_each} trials, objective={search_cfg.objective})")

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
            sid = f"loss_{center_id}_{trial.number:03d}"
            latent_path = out_dir / "candidates/latents" / f"{sid}_latent.npy"
            mask_path = out_dir / "candidates/masks" / f"{sid}_mask.npy"
            latent_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(latent_path, z_std)
            mask = decode_latent(latent_path, mask_path)
            if not check_mask_heuristic(mask).passed:
                return 1.0 + search_cfg.drc_penalty

            res = simulate_mask(mask, recipe, verbose=False)
            if res.status != "ok" or not np.isfinite(res.split_ratio_upper):
                return 3.0

            loss = meep_search_loss(res.split_ratio_upper, res.insertion_loss_db, search_cfg)
            se, il_pen, _ = pareto_tuple(
                res.split_ratio_upper,
                res.insertion_loss_db,
                target,
                max_il_db=search_cfg.max_insertion_loss_db,
            )
            split_err = abs(res.split_ratio_upper - target)
            trial.set_user_attr("sample_id", sid)
            trial.set_user_attr("R_up", res.split_ratio_upper)
            trial.set_user_attr("split_err", split_err)
            trial.set_user_attr("IL_db", res.insertion_loss_db)
            trial.set_user_attr("il_penalty", il_pen)
            trial.set_user_attr("in_spec", split_err <= tol)
            return loss

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=trials_each, show_progress_bar=True)

        for t in study.trials:
            if not t.user_attrs.get("sample_id"):
                continue
            rows.append(
                {
                    "center_id": center_id,
                    "sample_id": t.user_attrs["sample_id"],
                    "loss": t.value,
                    "R_up": t.user_attrs.get("R_up"),
                    "split_err": t.user_attrs.get("split_err"),
                    "IL_db": t.user_attrs.get("IL_db"),
                    "il_penalty": t.user_attrs.get("il_penalty"),
                    "in_spec": t.user_attrs.get("in_spec"),
                }
            )

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["loss", "split_err", "IL_db"], na_position="last")
    df.to_csv(out_dir / "loss_aware_trials.csv", index=False)

    summary = {
        "phase": "publication_b",
        "objective": search_cfg.objective,
        "weight_il": search_cfg.weight_il,
        "max_insertion_loss_db": search_cfg.max_insertion_loss_db,
        "n_trials": len(df),
        "n_in_spec": int(df["in_spec"].fillna(False).sum()) if len(df) else 0,
        "best_loss": float(df["loss"].min()) if len(df) else None,
        "best_split_err": float(df["split_err"].min()) if len(df) else None,
        "best_IL_db": float(df["IL_db"].min()) if len(df) else None,
        "output_dir": str(out_dir.relative_to(REPO)),
    }
    (out_dir / "loss_aware_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
