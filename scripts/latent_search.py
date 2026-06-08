#!/usr/bin/env python3
"""Bayesian optimization over latent space toward 50/50 split (surrogate-guided)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def resolve_repo_path(path: Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def as_repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())

from nano_inv.manifold import EBeamManifold  # noqa: E402
from nano_inv.search import SearchConfig, evaluate_candidate, make_objective  # noqa: E402
from nano_inv.latent import pad_latent_to_standard  # noqa: E402
from nano_inv.surrogate import load_artifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "phase0.yaml")
    p.add_argument(
        "--surrogate",
        type=Path,
        default=None,
        help="Path to surrogate.joblib (default: data/phase0/surrogate/surrogate.joblib)",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--top-k", type=int, default=20, help="Export top-k candidates by loss")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--mode",
        type=str,
        choices=["perturb", "perlin", "mixed"],
        default="mixed",
        help="Latent family to search",
    )
    p.add_argument("--no-drc", action="store_true", help="Do not penalize heuristic DRC failures")
    p.add_argument("--study-name", type=str, default="phase0_split_bo")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--surrogate-arch",
        type=str,
        default=None,
        help="Hint only (logged); use matching trained surrogate.joblib",
    )
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def trial_row(trial: optuna.trial.FrozenTrial, rank: int | None = None) -> dict:
    row = {
        "rank": rank,
        "trial_number": trial.number,
        "loss": trial.value,
        "pred_split_ratio_upper": trial.user_attrs.get("pred_split_ratio_upper"),
        "in_spec": trial.user_attrs.get("in_spec"),
        "family": trial.user_attrs.get("family"),
        "sigma": trial.user_attrs.get("sigma"),
        "scale": trial.user_attrs.get("scale"),
        "offset_x": trial.user_attrs.get("offset_x"),
        "offset_y": trial.user_attrs.get("offset_y"),
        "dim": trial.user_attrs.get("dim"),
        "drc_heuristic_pass": trial.user_attrs.get("drc_heuristic_pass"),
        "fill_ratio": trial.user_attrs.get("fill_ratio"),
        "drc_reasons": trial.user_attrs.get("drc_reasons"),
    }
    return row


def export_top_candidates(
    *,
    repo_root: Path,
    out_dir: Path,
    trials: list[optuna.trial.FrozenTrial],
    manifold: EBeamManifold,
    surrogate_path: Path,
    target: float,
    tolerance: float,
    seed: int,
) -> pd.DataFrame:
    cand_dir = out_dir / "candidates"
    latent_dir = cand_dir / "latents"
    mask_dir = cand_dir / "masks"
    latent_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for rank, trial in enumerate(trials, start=1):
        z, meta = _reconstruct_latent(trial, manifold.reference_latent, seed=seed)
        z = pad_latent_to_standard(np.asarray(z, dtype=np.float32))
        mask = manifold.decode_numpy(z)
        sample_id = f"search_{trial.number:05d}"
        latent_path = latent_dir / f"{sample_id}_latent.npy"
        mask_path = mask_dir / f"{sample_id}_mask.npy"
        np.save(latent_path, z)
        np.save(mask_path, mask)

        pred = trial.user_attrs.get("pred_split_ratio_upper")
        rows.append(
            {
                "rank": rank,
                "sample_id": sample_id,
                "trial_number": trial.number,
                "loss": trial.value,
                "pred_split_ratio_upper": pred,
                "in_spec_surrogate": trial.user_attrs.get("in_spec"),
                "latent_path": as_repo_relative(latent_path),
                "mask_path": as_repo_relative(mask_path),
                "surrogate_path": as_repo_relative(surrogate_path),
                "target_split_ratio": target,
                "tolerance": tolerance,
                **{k: v for k, v in meta.items()},
                "drc_heuristic_pass": trial.user_attrs.get("drc_heuristic_pass"),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "top_candidates.csv", index=False)
    return df


def _reconstruct_latent(
    trial: optuna.trial.FrozenTrial,
    reference: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed + trial.number)
    """Rebuild z from stored trial params (deterministic given trial number for perturb)."""
    from nano_inv.manifold import sample_latent_perlin, sample_latent_perturbation

    family = trial.user_attrs.get("family") or (
        "perturb" if trial.params.get("sigma") is not None else "perlin"
    )
    ref = np.asarray(reference, dtype=np.float32)
    if family == "perturb" or "sigma" in trial.params:
        sigma = float(trial.params["sigma"])
        z = sample_latent_perturbation(ref, rng, sigma=sigma)
        return z, {"family": "perturb", "sigma": sigma}
    scale = float(trial.params["scale"])
    offset_x = float(trial.params["offset_x"])
    offset_y = float(trial.params["offset_y"])
    dim = int(trial.params["dim"])
    z = sample_latent_perlin(scale=scale, offset=(offset_x, offset_y), dim=dim)
    return z, {
        "family": "perlin",
        "scale": scale,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "dim": dim,
    }


def main() -> None:
    args = parse_args()
    cfg_yaml = load_config(args.config)
    data_root = REPO_ROOT / cfg_yaml["data"]["root"]
    out_dir = resolve_repo_path(args.output_dir) if args.output_dir else data_root / "search"
    surrogate_path = resolve_repo_path(
        args.surrogate if args.surrogate else data_root / "surrogate" / "surrogate.joblib"
    )

    targets = cfg_yaml.get("targets") or {}
    target_split = float(targets.get("split_ratio_1550", 0.5))
    tolerance = float(targets.get("split_ratio_tolerance", 0.05))
    n_trials = args.n_trials or int((cfg_yaml.get("search") or {}).get("n_iterations", 200))

    if not surrogate_path.exists():
        raise SystemExit(
            f"surrogate not found: {surrogate_path}\n"
            "Run: python scripts/train_surrogate.py"
        )

    if args.dry_run:
        print(
            json.dumps(
                {
                    "surrogate": str(surrogate_path),
                    "n_trials": n_trials,
                    "mode": args.mode,
                    "target_split_ratio": target_split,
                    "tolerance": tolerance,
                    "output_dir": str(out_dir),
                },
                indent=2,
            )
        )
        return

    surrogate = load_artifact(surrogate_path)
    manifold = EBeamManifold.load()
    ref = np.asarray(manifold.reference_latent, dtype=np.float32)
    search_cfg = SearchConfig(
        target_split_ratio=target_split,
        tolerance=tolerance,
        require_drc=not args.no_drc,
        mode=args.mode,  # type: ignore[arg-type]
    )

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(
        make_objective(
            surrogate=surrogate,
            manifold=manifold,
            reference=ref,
            seed=args.seed,
            cfg=search_cfg,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    trials_df = pd.DataFrame([trial_row(t) for t in study.trials])
    trials_df.to_csv(out_dir / "search_trials.csv", index=False)

    completed = [t for t in study.trials if t.value is not None]
    completed.sort(key=lambda t: t.value)  # type: ignore[arg-type, return-value]
    top_trials = completed[: args.top_k]
    top_df = export_top_candidates(
        repo_root=REPO_ROOT,
        out_dir=out_dir,
        trials=top_trials,
        manifold=manifold,
        surrogate_path=surrogate_path,
        target=target_split,
        tolerance=tolerance,
        seed=args.seed,
    )

    n_in_spec = int(sum(1 for t in completed if t.user_attrs.get("in_spec")))
    best = study.best_trial
    summary = {
        "study_name": args.study_name,
        "n_trials": len(study.trials),
        "n_complete": len(completed),
        "n_in_spec_surrogate": n_in_spec,
        "best_trial": best.number,
        "best_loss": best.value,
        "best_pred_split_ratio_upper": best.user_attrs.get("pred_split_ratio_upper"),
        "target_split_ratio": target_split,
        "tolerance": tolerance,
        "mode": args.mode,
        "surrogate": as_repo_relative(surrogate_path),
        "output_dir": as_repo_relative(out_dir),
        "surrogate_architecture": getattr(surrogate, "architecture", "latent_mlp"),
        "top_k_exported": len(top_df),
    }
    (out_dir / "search_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nTop candidate for MEEP verify: {top_df.iloc[0]['sample_id'] if len(top_df) else 'none'}")
    print(
        "Verify with:\n"
        f"  bash scripts/run_meep.sh scripts/run_fdtd_batch.py "
        f"--manifest {as_repo_relative(out_dir)}/top_candidates.csv "
        f"--no-skip-existing"
    )


if __name__ == "__main__":
    main()
