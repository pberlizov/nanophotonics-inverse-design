#!/usr/bin/env python3
"""
B1 — MEEP-native search in latent space (not sigma-only).

Modes: residual (default), sigma, pca

  bash scripts/run_meep.sh scripts/latent_meep_search.py --latent-mode residual --n-trials 40
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
from nano_inv.latent import pad_latent_to_standard  # noqa: E402
from nano_inv.meep_latent import load_perturb_latents_from_manifest, suggest_latent_for_meep  # noqa: E402
from nano_inv.meep_sim import (  # noqa: E402
    MeepRecipe,
    require_meep,
    simulate_mask,
    simulate_mask_broadband,
)
from nano_inv.search_objectives import SearchConfig, broadband_split_loss, meep_search_loss  # noqa: E402


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
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/phase1_track_b.yaml")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--latent-mode", type=str, default=None, choices=["sigma", "residual", "pca", "perlin"])
    p.add_argument("--residual-dims", type=int, default=None)
    p.add_argument("--residual-bound", type=float, default=None)
    p.add_argument("--pca-dim", type=int, default=8)
    p.add_argument("--objective", type=str, default=None, choices=["split", "multi", "broadband"])
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    tb = cfg.get("track_b") or {}
    b1 = tb.get("latent_meep_search") or {}

    data_root = REPO_ROOT / cfg["data"]["root"]
    out_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / b1.get(
        "output_dir", "data/phase1/track_b/latent_meep_search"
    )
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = cfg.get("targets") or {}
    target = float(targets.get("split_ratio_1550", 0.5))
    tol = float(targets.get("split_ratio_tolerance", 0.05))
    meep_cfg = dict(cfg.get("meep") or {})
    version = meep_cfg.get("recipe_version", "phase0_v1")
    recipe = MeepRecipe.for_version(version, meep_cfg)

    bb_cfg = cfg.get("broadband") or {}
    if bb_cfg.get("wl_start") is not None:
        wl_start = float(bb_cfg["wl_start"])
        wl_stop = float(bb_cfg["wl_stop"])
        wl_step = float(bb_cfg.get("wl_step", 0.01))
        wavelengths = list(np.arange(wl_start, wl_stop + 0.5 * wl_step, wl_step))
    else:
        wavelengths = [float(x) for x in (tb.get("broadband_wavelengths_um") or [1.50, 1.55, 1.60])]
    objective_mode = args.objective or b1.get("objective", "multi")
    latent_mode = args.latent_mode or b1.get("latent_mode", "residual")

    search_block = cfg.get("search") or {}
    search_cfg = SearchConfig(
        target_split_ratio=target,
        tolerance=tol,
        objective="multi" if objective_mode == "broadband" else objective_mode,
        max_insertion_loss_db=float(targets.get("max_insertion_loss_db", 12.0)),
        weight_split=float(search_block.get("weight_split", 1.0)),
        weight_il=float(search_block.get("weight_il", 0.15)),
    )

    require_meep()
    ref = np.load(data_root / "latents" / "ref_published_latent.npy").astype(np.float32)

    pca_bundle = None
    if latent_mode == "pca":
        for pca_path in (
            out_dir / "pca_basis.npz",
            REPO_ROOT / "data/phase1/track_b/latent_pca_basis.npz",
        ):
            if pca_path.exists():
                data = np.load(pca_path)
                pca_bundle = (data["mean_flat"], data["components"])
                break
        if pca_bundle is None:
            raise SystemExit(
                "PCA basis missing. Run in .venv:\n"
                "  python scripts/fit_latent_pca_basis.py"
            )

    residual_dims = args.residual_dims or int(b1.get("residual_dims", 12))
    residual_bound = args.residual_bound or float(b1.get("residual_bound", 0.06))

    def objective(trial: optuna.Trial) -> float:
        rng = np.random.default_rng(args.seed + trial.number)
        z, meta = suggest_latent_for_meep(
            trial,
            ref,
            rng,
            mode=latent_mode,
            residual_dims=residual_dims,
            residual_bound=residual_bound,
            pca_bundle=pca_bundle,
        )
        z_std = pad_latent_to_standard(z)
        sid = f"latent_meep_{trial.number:05d}"
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

        if objective_mode == "broadband":
            bb = simulate_mask_broadband(
                mask, recipe, wavelengths, target_split=target, verbose=args.verbose and trial.number < 1
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
            trial.set_user_attr("splits_by_wavelength", bb.splits_by_wavelength)
            trial.set_user_attr("meep_split_ratio_upper", bb.splits_by_wavelength.get(1.55, np.nan))
            trial.set_user_attr("insertion_loss_db", bb.mean_insertion_loss_db)
        else:
            res = simulate_mask(mask, recipe, verbose=args.verbose and trial.number < 2)
            if res.status != "ok" or not np.isfinite(res.split_ratio_upper):
                return 2.0
            loss = meep_search_loss(res.split_ratio_upper, res.insertion_loss_db, search_cfg)
            trial.set_user_attr("meep_split_ratio_upper", res.split_ratio_upper)
            trial.set_user_attr("insertion_loss_db", res.insertion_loss_db)

        trial.set_user_attr("in_spec", abs(trial.user_attrs.get("meep_split_ratio_upper", 99) - target) <= tol)
        trial.set_user_attr("sample_id", sid)
        for k, v in meta.items():
            trial.set_user_attr(k, v)
        return float(loss)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        study_name=f"latent_meep_{latent_mode}_{version}",
    )
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    rows = [
        {
            "trial_number": t.number,
            "loss": t.value,
            "meep_split_ratio_upper": t.user_attrs.get("meep_split_ratio_upper"),
            "insertion_loss_db": t.user_attrs.get("insertion_loss_db"),
            "in_spec": t.user_attrs.get("in_spec"),
            "latent_mode": t.user_attrs.get("latent_mode"),
            "sample_id": t.user_attrs.get("sample_id"),
        }
        for t in study.trials
    ]
    pd.DataFrame(rows).to_csv(out_dir / "latent_meep_trials.csv", index=False)

    completed = sorted(
        [t for t in study.trials if t.user_attrs.get("meep_split_ratio_upper") is not None],
        key=lambda t: t.value or 99,
    )
    cand_rows = []
    for rank, t in enumerate(completed[: args.top_k], start=1):
        sid = t.user_attrs["sample_id"]
        cand_rows.append(
            {
                "rank": rank,
                "sample_id": sid,
                "trial_number": t.number,
                "loss": t.value,
                "split_ratio_upper": t.user_attrs.get("meep_split_ratio_upper"),
                "in_spec": t.user_attrs.get("in_spec"),
                "latent_mode": latent_mode,
                "latent_path": str((out_dir / "candidates/latents" / f"{sid}_latent.npy").relative_to(REPO_ROOT)),
                "mask_path": str((out_dir / "candidates/masks" / f"{sid}_mask.npy").relative_to(REPO_ROOT)),
                "recipe_version": version,
            }
        )
    pd.DataFrame(cand_rows).to_csv(out_dir / "top_candidates.csv", index=False)

    best = study.best_trial
    summary = {
        "latent_mode": latent_mode,
        "objective": objective_mode,
        "recipe_version": version,
        "n_trials": len(study.trials),
        "best_trial": best.number,
        "best_meep_split": best.user_attrs.get("meep_split_ratio_upper"),
        "best_sample_id": best.user_attrs.get("sample_id"),
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
    }
    (out_dir / "latent_meep_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
