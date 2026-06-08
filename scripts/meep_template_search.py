#!/usr/bin/env python3
"""
B3 — Co-search mask perturbation (sigma) and MEEP template geometry params.

  bash scripts/run_meep.sh scripts/meep_template_search.py --n-trials 30
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = REPO_ROOT / ".venv/bin/python"
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.drc_heuristic import check_mask_heuristic  # noqa: E402
from nano_inv.latent import pad_latent_to_standard, sample_latent_perturbation  # noqa: E402
from nano_inv.meep_sim import MeepRecipe, require_meep, simulate_mask  # noqa: E402
from nano_inv.search_objectives import SearchConfig, meep_search_loss  # noqa: E402


def decode_latent_to_mask(latent_path: Path, mask_path: Path) -> np.ndarray:
    subprocess.run(
        [str(VENV_PY), str(REPO_ROOT / "scripts/decode_one.py"), "--latent", str(latent_path), "--mask", str(mask_path)],
        cwd=REPO_ROOT,
        check=True,
    )
    return np.load(mask_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/phase1_track_b.yaml")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    tb = cfg.get("track_b") or {}
    b3 = tb.get("template_search") or {}

    data_root = REPO_ROOT / cfg["data"]["root"]
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / b3.get(
        "output_dir", "data/phase1/track_b/template_search"
    )
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = cfg.get("targets") or {}
    target = float(targets.get("split_ratio_1550", 0.5))
    tol = float(targets.get("split_ratio_tolerance", 0.05))
    meep_cfg = dict(cfg.get("meep") or {})
    base_recipe = MeepRecipe.for_version(meep_cfg.get("recipe_version", "phase0_v1"), meep_cfg)

    tpl = b3.get("template_bounds") or {}
    wg_bounds = tpl.get("wg_width_um", [0.38, 0.48])
    arm_bounds = tpl.get("arm_y_upper", [0.50, 0.65])

    search_cfg = SearchConfig(
        target_split_ratio=target,
        tolerance=tol,
        objective=cfg.get("search", {}).get("objective", "multi"),
        max_insertion_loss_db=float(targets.get("max_insertion_loss_db", 12.0)),
    )

    require_meep()
    ref = np.load(data_root / "latents" / "ref_published_latent.npy").astype(np.float32)

    def objective(trial: optuna.Trial) -> float:
        rng = np.random.default_rng(args.seed + trial.number)
        sigma = trial.suggest_float("sigma", 0.008, 0.04, log=True)
        wg = trial.suggest_float("wg_width_um", float(wg_bounds[0]), float(wg_bounds[1]))
        arm_u = trial.suggest_float("arm_y_upper", float(arm_bounds[0]), float(arm_bounds[1]))
        recipe = replace(
            base_recipe,
            wg_width_um=wg,
            arm_y_upper=arm_u,
            arm_y_lower=-arm_u,
        )

        z = pad_latent_to_standard(sample_latent_perturbation(ref, rng, sigma=sigma))
        sid = f"tpl_meep_{trial.number:05d}"
        latent_path = out_dir / "candidates/latents" / f"{sid}_latent.npy"
        mask_path = out_dir / "candidates/masks" / f"{sid}_mask.npy"
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(latent_path, z)
        mask = decode_latent_to_mask(latent_path, mask_path)
        if not check_mask_heuristic(mask).passed:
            return 1.0 + search_cfg.drc_penalty

        res = simulate_mask(mask, recipe, verbose=False)
        if res.status != "ok":
            return 2.0
        trial.set_user_attr("wg_width_um", wg)
        trial.set_user_attr("arm_y_upper", arm_u)
        trial.set_user_attr("sigma", sigma)
        trial.set_user_attr("meep_split_ratio_upper", res.split_ratio_upper)
        trial.set_user_attr("insertion_loss_db", res.insertion_loss_db)
        trial.set_user_attr("in_spec", abs(res.split_ratio_upper - target) <= tol)
        trial.set_user_attr("sample_id", sid)
        return meep_search_loss(res.split_ratio_upper, res.insertion_loss_db, search_cfg)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    completed = sorted(
        [t for t in study.trials if t.user_attrs.get("meep_split_ratio_upper") is not None],
        key=lambda t: t.value or 99,
    )
    rows = []
    for rank, t in enumerate(completed[: args.top_k], start=1):
        sid = t.user_attrs["sample_id"]
        rows.append(
            {
                "rank": rank,
                "sample_id": sid,
                "split_ratio_upper": t.user_attrs["meep_split_ratio_upper"],
                "wg_width_um": t.user_attrs.get("wg_width_um"),
                "arm_y_upper": t.user_attrs.get("arm_y_upper"),
                "sigma": t.params.get("sigma"),
                "latent_path": str((out_dir / f"candidates/latents/{sid}_latent.npy").relative_to(REPO_ROOT)),
                "mask_path": str((out_dir / f"candidates/masks/{sid}_mask.npy").relative_to(REPO_ROOT)),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "top_candidates.csv", index=False)
    summary = {
        "n_trials": args.n_trials,
        "best_trial": study.best_trial.number,
        "best_meep_split": study.best_trial.user_attrs.get("meep_split_ratio_upper"),
        "best_wg_width_um": study.best_trial.user_attrs.get("wg_width_um"),
        "best_arm_y_upper": study.best_trial.user_attrs.get("arm_y_upper"),
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
    }
    (out_dir / "template_search_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
