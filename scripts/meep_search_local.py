#!/usr/bin/env python3
"""
Local MEEP search: refine sigma around a champion perturbation (phase0_v1).

  bash scripts/run_meep.sh scripts/meep_search_local.py \
    --center-sigma 0.0226 --n-trials 50
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
from nano_inv.meep_sim import (  # noqa: E402
    MeepRecipe,
    require_meep,
    simulate_mask,
    simulate_mask_broadband,
)
from nano_inv.search_objectives import (  # noqa: E402
    SearchConfig,
    broadband_split_loss,
    meep_search_loss,
)


def decode_latent_to_mask(latent_path: Path, mask_path: Path) -> np.ndarray:
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
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/phase1/meep_search_local")
    p.add_argument("--center-sigma", type=float, default=0.022620917081796706)
    p.add_argument("--sigma-min", type=float, default=None)
    p.add_argument("--sigma-max", type=float, default=None)
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--objective",
        type=str,
        default=None,
        choices=["split", "multi", "broadband"],
        help="split=1550 nm only; broadband=worst-case split over C-band grid",
    )
    p.add_argument("--broadband-wl-start", type=float, default=None)
    p.add_argument("--broadband-wl-stop", type=float, default=None)
    p.add_argument("--broadband-wl-step", type=float, default=None)
    return p.parse_args()


def c_band_wavelengths(
    cfg: dict,
    *,
    wl_start: float | None,
    wl_stop: float | None,
    wl_step: float | None,
) -> list[float]:
    bb = cfg.get("broadband") or {}
    start = wl_start if wl_start is not None else float(bb.get("wl_start", 1.53))
    stop = wl_stop if wl_stop is not None else float(bb.get("wl_stop", 1.57))
    step = wl_step if wl_step is not None else float(bb.get("wl_step", 0.01))
    wls = [round(float(w), 3) for w in np.arange(start, stop + 1e-9, step)]
    return wls


def main() -> None:
    args = parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    data_root = REPO_ROOT / cfg["data"]["root"]
    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    center = args.center_sigma
    s_min = args.sigma_min if args.sigma_min is not None else max(0.01, center * 0.5)
    s_max = args.sigma_max if args.sigma_max is not None else min(0.2, center * 2.5)

    targets = cfg.get("targets") or {}
    target = float(targets.get("split_ratio_1550", 0.5))
    tol = float(targets.get("split_ratio_tolerance", 0.05))
    meep_cfg = dict(cfg.get("meep") or {})
    recipe = MeepRecipe.for_version(meep_cfg.get("recipe_version", "phase0_v1"), meep_cfg)

    require_meep()
    ref = np.load(data_root / "latents" / "ref_published_latent.npy").astype(np.float32)
    search_block = cfg.get("search") or {}
    objective_mode = args.objective or search_block.get("objective", "split")
    wavelengths = c_band_wavelengths(
        cfg,
        wl_start=args.broadband_wl_start,
        wl_stop=args.broadband_wl_stop,
        wl_step=args.broadband_wl_step,
    )
    search_cfg = SearchConfig(
        target_split_ratio=target,
        tolerance=tol,
        objective="multi" if objective_mode == "broadband" else objective_mode,
        max_insertion_loss_db=float(targets.get("max_insertion_loss_db", 12.0)),
        weight_split=float(search_block.get("weight_split", 1.0)),
        weight_il=float(search_block.get("weight_il", 0.15)),
        drc_penalty=float(search_block.get("drc_penalty", 1.0)),
    )
    bb_gate = float((cfg.get("broadband") or {}).get("max_worst_split_error", tol))

    def objective(trial: optuna.Trial) -> float:
        rng = np.random.default_rng(args.seed + trial.number)
        sigma = trial.suggest_float("sigma", s_min, s_max, log=True)
        z = sample_latent_perturbation(ref, rng, sigma=sigma)
        z_std = pad_latent_to_standard(z)
        sid = f"local_{trial.number:05d}"
        latent_path = out_dir / "candidates" / "latents" / f"{sid}_latent.npy"
        mask_path = out_dir / "candidates" / "masks" / f"{sid}_mask.npy"
        latent_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(latent_path, z_std)
        mask = decode_latent_to_mask(latent_path, mask_path)
        if not check_mask_heuristic(mask).passed:
            return 1.0 + search_cfg.drc_penalty
        if objective_mode == "broadband":
            bb = simulate_mask_broadband(
                mask,
                recipe,
                wavelengths,
                target_split=target,
                verbose=args.verbose and trial.number < 1,
            )
            if bb.status != "ok":
                return 2.0
            loss = broadband_split_loss(
                bb.splits_by_wavelength,
                target,
                flatness_weight=float(search_block.get("flatness_weight", 0.0)),
            )
            loss += search_cfg.weight_il * max(
                0.0, bb.mean_insertion_loss_db - search_cfg.max_insertion_loss_db
            ) / max(search_cfg.max_insertion_loss_db, 1e-6)
            split_155 = bb.splits_by_wavelength.get(1.55, float("nan"))
            trial.set_user_attr("splits_by_wavelength", {str(k): v for k, v in bb.splits_by_wavelength.items()})
            trial.set_user_attr("worst_split_error", bb.worst_split_error)
            trial.set_user_attr("meep_split_ratio_upper", split_155)
            trial.set_user_attr("insertion_loss_db", bb.mean_insertion_loss_db)
            trial.set_user_attr("in_spec", bb.worst_split_error <= bb_gate)
            trial.set_user_attr("broadband_pass", bb.worst_split_error <= bb_gate)
        else:
            res = simulate_mask(mask, recipe, verbose=args.verbose and trial.number < 2)
            if res.status != "ok" or not np.isfinite(res.split_ratio_upper):
                return 2.0
            loss = meep_search_loss(res.split_ratio_upper, res.insertion_loss_db, search_cfg)
            trial.set_user_attr("meep_split_ratio_upper", res.split_ratio_upper)
            trial.set_user_attr("insertion_loss_db", res.insertion_loss_db)
            trial.set_user_attr("in_spec", abs(res.split_ratio_upper - target) <= tol)
        trial.set_user_attr("sample_id", sid)
        return loss

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    rows = [
        {
            "trial_number": t.number,
            "loss": t.value,
            "sigma": t.params.get("sigma"),
            "meep_split_ratio_upper": t.user_attrs.get("meep_split_ratio_upper"),
            "worst_split_error": t.user_attrs.get("worst_split_error"),
            "broadband_pass": t.user_attrs.get("broadband_pass"),
            "in_spec": t.user_attrs.get("in_spec"),
            "sample_id": t.user_attrs.get("sample_id"),
        }
        for t in study.trials
    ]
    pd.DataFrame(rows).to_csv(out_dir / "meep_search_local_trials.csv", index=False)

    completed = sorted(
        [t for t in study.trials if t.user_attrs.get("meep_split_ratio_upper") is not None],
        key=lambda t: t.value or 99,
    )
    top = completed[: args.top_k]
    cand = [
        {
            "rank": i + 1,
            "sample_id": t.user_attrs["sample_id"],
            "sigma": t.params["sigma"],
            "split_ratio_upper": t.user_attrs["meep_split_ratio_upper"],
            "in_spec": t.user_attrs["in_spec"],
            "latent_path": str(
                (out_dir / "candidates/latents" / f"{t.user_attrs['sample_id']}_latent.npy").relative_to(
                    REPO_ROOT
                )
            ),
            "mask_path": str(
                (out_dir / "candidates/masks" / f"{t.user_attrs['sample_id']}_mask.npy").relative_to(
                    REPO_ROOT
                )
            ),
            "recipe_version": meep_cfg.get("recipe_version", "phase0_v1"),
        }
        for i, t in enumerate(top)
    ]
    pd.DataFrame(cand).to_csv(out_dir / "top_candidates.csv", index=False)

    best = study.best_trial
    summary = {
        "mode": "local_sigma_refinement",
        "objective": objective_mode,
        "broadband_wavelengths_um": wavelengths if objective_mode == "broadband" else None,
        "center_sigma": center,
        "sigma_range": [s_min, s_max],
        "n_trials": len(study.trials),
        "n_in_spec": int(sum(1 for t in completed if t.user_attrs.get("in_spec"))),
        "n_broadband_pass": int(sum(1 for t in completed if t.user_attrs.get("broadband_pass"))),
        "best_trial": best.number,
        "best_meep_split": best.user_attrs.get("meep_split_ratio_upper"),
        "best_sigma": best.params.get("sigma"),
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
    }
    (out_dir / "meep_search_local_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
