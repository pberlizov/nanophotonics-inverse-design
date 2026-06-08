#!/usr/bin/env python3
"""
MEEP-native Bayesian search: objective = real FDTD split ratio (no surrogate).

Decode uses .venv (drcgenerator); MEEP + optuna use env mp via run_meep.sh.

  bash scripts/run_meep.sh scripts/meep_search.py --n-trials 30
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

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = REPO_ROOT / ".venv/bin/python"
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.drc_heuristic import check_mask_heuristic  # noqa: E402
from nano_inv.latent import pad_latent_to_standard, sample_latent_perturbation  # noqa: E402
from nano_inv.meep_sim import MeepRecipe, require_meep, simulate_mask  # noqa: E402
from nano_inv.search_objectives import SearchConfig, meep_search_loss  # noqa: E402


def decode_latent_to_mask(latent_path: Path, mask_path: Path) -> np.ndarray:
    """Run drcgenerator decode in project .venv (not mp)."""
    if not VENV_PY.exists():
        raise SystemExit(f"missing {VENV_PY} — run: bash scripts/setup.sh")
    subprocess.run(
        [
            str(VENV_PY),
            str(REPO_ROOT / "scripts/decode_one.py"),
            "--latent",
            str(latent_path),
            "--mask",
            str(mask_path),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    return np.load(mask_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase0.yaml")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--recipe-version", type=str, default=None)
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--objective", type=str, default=None, choices=["split", "multi"])
    p.add_argument("--sigma-min", type=float, default=None)
    p.add_argument("--sigma-max", type=float, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_root = REPO_ROOT / cfg["data"]["root"]
    out_dir = Path(args.output_dir) if args.output_dir else data_root / "meep_search"
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = cfg.get("targets") or {}
    target = float(targets.get("split_ratio_1550", 0.5))
    tol = float(targets.get("split_ratio_tolerance", 0.05))

    meep_cfg = dict(cfg.get("meep") or {})
    if args.resolution:
        meep_cfg["resolution"] = args.resolution
    version = args.recipe_version or meep_cfg.get("recipe_version", "phase0_v1")
    recipe = MeepRecipe.for_version(version, meep_cfg)

    require_meep()
    ref_path = data_root / "latents" / "ref_published_latent.npy"
    if not ref_path.exists():
        raise SystemExit(f"missing {ref_path} — run decode_batch.py first")
    ref = np.load(ref_path).astype(np.float32)
    search_block = cfg.get("search") or {}
    objective_mode = args.objective or search_block.get("objective", "split")
    max_il = float(targets.get("max_insertion_loss_db", 12.0))
    search_cfg = SearchConfig(
        target_split_ratio=target,
        tolerance=tol,
        objective=objective_mode,
        max_insertion_loss_db=max_il,
        weight_split=float(search_block.get("weight_split", 1.0)),
        weight_il=float(search_block.get("weight_il", 0.15)),
        drc_penalty=float(search_block.get("drc_penalty", 1.0)),
    )
    s_min = args.sigma_min if args.sigma_min is not None else float(search_block.get("sigma_min", 0.02))
    s_max = args.sigma_max if args.sigma_max is not None else float(search_block.get("sigma_max", 0.20))

    def objective(trial: optuna.Trial) -> float:
        rng = np.random.default_rng(args.seed + trial.number)
        sigma = trial.suggest_float("sigma", s_min, s_max, log=True)
        z = sample_latent_perturbation(ref, rng, sigma=sigma)
        z_std = pad_latent_to_standard(z)
        sid = f"meep_bo_{trial.number:05d}"
        latent_dir = out_dir / "candidates" / "latents"
        mask_dir = out_dir / "candidates" / "masks"
        latent_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        latent_path = latent_dir / f"{sid}_latent.npy"
        mask_path = mask_dir / f"{sid}_mask.npy"
        np.save(latent_path, z_std)
        mask = decode_latent_to_mask(latent_path, mask_path)
        drc = check_mask_heuristic(mask)
        if not drc.passed:
            return 1.0 + search_cfg.drc_penalty

        res = simulate_mask(mask, recipe, verbose=args.verbose and trial.number < 2)
        if res.status != "ok" or not np.isfinite(res.split_ratio_upper):
            trial.set_user_attr("status", res.status)
            trial.set_user_attr("error", res.error)
            return 2.0

        loss = meep_search_loss(res.split_ratio_upper, res.insertion_loss_db, search_cfg)
        trial.set_user_attr("meep_split_ratio_upper", res.split_ratio_upper)
        trial.set_user_attr("insertion_loss_db", res.insertion_loss_db)
        trial.set_user_attr("objective", objective_mode)
        trial.set_user_attr("sigma", sigma)
        trial.set_user_attr("in_spec", abs(res.split_ratio_upper - target) <= tol)
        trial.set_user_attr("drc_heuristic_pass", drc.passed)
        trial.set_user_attr("sample_id", sid)
        return loss

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        study_name=f"meep_perturb_{version}",
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    rows = []
    for t in study.trials:
        rows.append(
            {
                "trial_number": t.number,
                "loss": t.value,
                "meep_split_ratio_upper": t.user_attrs.get("meep_split_ratio_upper"),
                "in_spec": t.user_attrs.get("in_spec"),
                "sigma": t.params.get("sigma"),
                "sample_id": t.user_attrs.get("sample_id"),
                "status": t.user_attrs.get("status", "ok"),
            }
        )
    trials_df = pd.DataFrame(rows)
    trials_df.to_csv(out_dir / "meep_search_trials.csv", index=False)

    completed = [t for t in study.trials if t.value is not None and t.user_attrs.get("meep_split_ratio_upper") is not None]
    completed.sort(key=lambda t: t.value)  # type: ignore[arg-type, return-value]
    top = completed[: args.top_k]

    cand_rows = []
    for rank, t in enumerate(top, start=1):
        sid = t.user_attrs.get("sample_id", f"meep_bo_{t.number:05d}")
        cand_rows.append(
            {
                "rank": rank,
                "sample_id": sid,
                "trial_number": t.number,
                "loss": t.value,
                "split_ratio_upper": t.user_attrs.get("meep_split_ratio_upper"),
                "in_spec": t.user_attrs.get("in_spec"),
                "sigma": t.params.get("sigma"),
                "latent_path": str((out_dir / "candidates" / "latents" / f"{sid}_latent.npy").relative_to(REPO_ROOT)),
                "mask_path": str((out_dir / "candidates" / "masks" / f"{sid}_mask.npy").relative_to(REPO_ROOT)),
                "recipe_version": version,
            }
        )
    top_df = pd.DataFrame(cand_rows)
    top_df.to_csv(out_dir / "top_candidates.csv", index=False)

    best = study.best_trial
    summary = {
        "recipe_version": version,
        "objective": objective_mode,
        "n_trials": len(study.trials),
        "n_complete": len(completed),
        "n_in_spec_meep": int(sum(1 for t in completed if t.user_attrs.get("in_spec"))),
        "best_trial": best.number,
        "best_meep_split": best.user_attrs.get("meep_split_ratio_upper"),
        "best_sigma": best.params.get("sigma"),
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
    }
    (out_dir / "meep_search_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
