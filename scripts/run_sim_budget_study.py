#!/usr/bin/env python3
"""
Tier 1 — Sim-budget study: compare policies at equal MEEP budget B.

Run under mp for MEEP policies:
  bash scripts/run_meep.sh scripts/run_sim_budget_study.py --pilot
  bash scripts/run_meep.sh scripts/run_sim_budget_study.py --budget 50 --policy sigma_meep

Generate ranked candidates first (.venv):
  python scripts/generate_ranked_candidates.py --n-proposals 500 --output data/phase1/wedge_a/sim_budget/candidates_pool.csv
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
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = REPO_ROOT / ".venv/bin/python"
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.drc_heuristic import check_mask_heuristic  # noqa: E402
from nano_inv.latent import pad_latent_to_standard, sample_latent_perturbation  # noqa: E402
from nano_inv.meep_sim import MeepRecipe, require_meep, simulate_mask  # noqa: E402
from nano_inv.pilot import load_pilot_config  # noqa: E402
from nano_inv.search_objectives import SearchConfig, meep_search_loss  # noqa: E402
from nano_inv.sim_budget import MeepRow, is_in_spec, rows_to_dataframe, summarize_meep_rows  # noqa: E402


def decode_latent_to_mask(latent_path: Path, mask_path: Path) -> np.ndarray:
    subprocess.run(
        [str(VENV_PY), str(REPO_ROOT / "scripts/decode_one.py"), "--latent", str(latent_path), "--mask", str(mask_path)],
        cwd=REPO_ROOT,
        check=True,
    )
    return np.load(mask_path)


def meep_one(mask: np.ndarray, recipe: MeepRecipe, search_cfg: SearchConfig) -> tuple[float, float, float]:
    res = simulate_mask(mask, recipe, verbose=False)
    if res.status != "ok" or not np.isfinite(res.split_ratio_upper):
        return float("nan"), float("nan"), 2.0
    loss = meep_search_loss(res.split_ratio_upper, res.insertion_loss_db, search_cfg)
    return res.split_ratio_upper, res.insertion_loss_db, loss


def policy_random_perturb(
    budget: int,
    *,
    ref: np.ndarray,
    recipe: MeepRecipe,
    target: float,
    tol: float,
    search_cfg: SearchConfig,
    rng: np.random.Generator,
    s_min: float,
    s_max: float,
    out_dir: Path,
    centers: list[np.ndarray] | None = None,
) -> list[MeepRow]:
    rows: list[MeepRow] = []
    latent_dir = out_dir / "candidates/latents"
    mask_dir = out_dir / "candidates/masks"
    latent_dir.mkdir(parents=True, exist_ok=True)
    center_list = centers if centers else [ref]
    for i in tqdm(range(budget), desc="random_perturb"):
        sigma = float(rng.uniform(s_min, s_max))
        center = center_list[int(rng.integers(0, len(center_list)))]
        z = pad_latent_to_standard(sample_latent_perturbation(center, rng, sigma=sigma))
        sid = f"rnd_{budget:03d}_{i:05d}"
        lp, mp = latent_dir / f"{sid}_latent.npy", mask_dir / f"{sid}_mask.npy"
        np.save(lp, z)
        mask = decode_latent_to_mask(lp, mp)
        if not check_mask_heuristic(mask).passed:
            continue
        split, il, _ = meep_one(mask, recipe, search_cfg)
        rows.append(
            MeepRow(
                sample_id=sid,
                split_ratio_upper=split,
                insertion_loss_db=il,
                in_spec=is_in_spec(split, target, tol),
                sigma=sigma,
                policy="random_perturb",
                meep_budget=budget,
            )
        )
    return rows


def policy_sigma_meep(
    budget: int,
    *,
    ref: np.ndarray,
    recipe: MeepRecipe,
    target: float,
    tol: float,
    search_cfg: SearchConfig,
    seed: int,
    s_min: float,
    s_max: float,
    out_dir: Path,
) -> list[MeepRow]:
    latent_dir = out_dir / "candidates/latents"
    mask_dir = out_dir / "candidates/masks"
    latent_dir.mkdir(parents=True, exist_ok=True)
    trial_rows: list[MeepRow] = []

    def objective(trial: optuna.Trial) -> float:
        rng = np.random.default_rng(seed + trial.number)
        sigma = trial.suggest_float("sigma", s_min, s_max, log=True)
        z = pad_latent_to_standard(sample_latent_perturbation(ref, rng, sigma=sigma))
        sid = f"sig_{budget:03d}_{trial.number:05d}"
        lp, mp = latent_dir / f"{sid}_latent.npy", mask_dir / f"{sid}_mask.npy"
        np.save(lp, z)
        mask = decode_latent_to_mask(lp, mp)
        if not check_mask_heuristic(mask).passed:
            return 1.0 + search_cfg.drc_penalty
        split, il, loss = meep_one(mask, recipe, search_cfg)
        trial_rows.append(
            MeepRow(
                sample_id=sid,
                split_ratio_upper=split,
                insertion_loss_db=il,
                in_spec=is_in_spec(split, target, tol),
                sigma=sigma,
                policy="sigma_meep",
                meep_budget=budget,
            )
        )
        trial.set_user_attr("meep_split", split)
        return loss

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=budget, show_progress_bar=True)
    return trial_rows


def load_mask_for_candidate(row: pd.Series) -> np.ndarray:
    mask_rel = str(row.get("mask_path", "") or "").strip()
    latent_rel = str(row.get("latent_path", "") or "").strip()
    if mask_rel:
        mask_path = REPO_ROOT / mask_rel
        if mask_path.is_file():
            return np.load(mask_path)
    if not latent_rel:
        raise FileNotFoundError("candidate row missing mask_path and latent_path")
    lp = REPO_ROOT / latent_rel
    tmp = lp.parent / f"_decode_{lp.stem}.npy"
    return decode_latent_to_mask(lp, tmp)


def policy_surrogate_rank(
    budget: int,
    *,
    candidates_csv: Path,
    recipe: MeepRecipe,
    target: float,
    tol: float,
    search_cfg: SearchConfig,
    out_dir: Path,
    diverse_top_k: bool = False,
    diverse_min_hamming: float = 0.02,
) -> list[MeepRow]:
    df = pd.read_csv(candidates_csv)
    if diverse_top_k:
        from nano_inv.candidate_pool import diverse_top_k as select_diverse

        top = select_diverse(
            df, budget, repo_root=REPO_ROOT, min_hamming=diverse_min_hamming
        )
    else:
        top = df.head(budget)
    rows: list[MeepRow] = []
    for _, r in tqdm(top.iterrows(), total=len(top), desc="surrogate_rank"):
        mask = load_mask_for_candidate(r)
        split, il, _ = meep_one(mask, recipe, search_cfg)
        rows.append(
            MeepRow(
                sample_id=str(r["sample_id"]),
                split_ratio_upper=split,
                insertion_loss_db=il,
                in_spec=is_in_spec(split, target, tol),
                sigma=float(r.get("sigma", np.nan)),
                policy="surrogate_rank",
                meep_budget=budget,
            )
        )
    return rows


def parse_hierarchical_frac(policy: str, default_frac: float) -> float:
    """hierarchical_35 → 0.35, hierarchical_50 or hierarchical → default."""
    if policy in ("hierarchical", "hierarchical_50"):
        return default_frac
    if policy.startswith("hierarchical_"):
        try:
            return int(policy.split("_", 1)[1]) / 100.0
        except ValueError:
            pass
    return default_frac


def policy_hierarchical(
    budget: int,
    *,
    ref: np.ndarray,
    recipe: MeepRecipe,
    target: float,
    tol: float,
    search_cfg: SearchConfig,
    seed: int,
    s_min: float,
    s_max: float,
    cfg_path: Path,
    sur_path: Path,
    out_dir: Path,
    mult: int,
    frac: float,
    sigma_span: float = 0.012,
    policy_label: str = "hierarchical",
    diverse_top_k: bool = False,
    diverse_min_hamming: float = 0.02,
) -> list[MeepRow]:
    n_sigma = max(1, int(budget * frac))
    n_rank = max(1, budget - n_sigma)
    sigma_rows = policy_sigma_meep(
        n_sigma, ref=ref, recipe=recipe, target=target, tol=tol, search_cfg=search_cfg,
        seed=seed, s_min=s_min, s_max=s_max, out_dir=out_dir / "phase_sigma",
    )
    best_sigma = 0.02
    if sigma_rows:
        valid = [r for r in sigma_rows if np.isfinite(r.split_ratio_upper)]
        if valid:
            best = min(valid, key=lambda r: abs(r.split_ratio_upper - target))
            best_sigma = float(best.sigma or 0.02)

    cand_csv = out_dir / f"candidates_hier_{budget}.csv"
    subprocess.run(
        [
            str(VENV_PY),
            "scripts/generate_ranked_candidates.py",
            "--config",
            str(cfg_path),
            "--surrogate",
            str(sur_path.relative_to(REPO_ROOT)),
            "--n-proposals",
            str(max(n_rank * mult, 100)),
            "--output",
            str(cand_csv.relative_to(REPO_ROOT)),
            "--center-sigma",
            str(best_sigma),
            "--sigma-span",
            str(sigma_span),
            "--seed",
            str(seed + 1),
            "--latent-only",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    rank_rows = policy_surrogate_rank(
        n_rank,
        candidates_csv=cand_csv,
        recipe=recipe,
        target=target,
        tol=tol,
        search_cfg=search_cfg,
        out_dir=out_dir / "phase_rank",
        diverse_top_k=diverse_top_k,
        diverse_min_hamming=diverse_min_hamming,
    )
    for r in rank_rows:
        r.policy = policy_label
    for r in sigma_rows:
        r.policy = policy_label
    return sigma_rows + rank_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/wedge_a.yaml")
    p.add_argument("--policy", type=str, default="all", help="all | random_perturb | sigma_meep | surrogate_rank | hierarchical")
    p.add_argument("--budget", type=int, default=None)
    p.add_argument("--pilot", action="store_true")
    p.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=None,
        help="Override budget list, e.g. --budgets 50 100",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ensure-candidates", action="store_true", help="Regenerate candidate pool via .venv")
    p.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip policy/budget if meep_results.csv exists with >= budget rows",
    )
    p.add_argument(
        "--replace-run",
        action="store_true",
        help="Replace last run in wedge_a_metrics.json instead of appending",
    )
    p.add_argument(
        "--replicate-id",
        type=int,
        default=None,
        help="Replication run id (uses sim_budget/replicates/run_XX/)",
    )
    p.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Override metrics JSON path",
    )
    return p.parse_args()


def run_is_complete(run_dir: Path, budget: int) -> bool:
    csv_path = run_dir / "meep_results.csv"
    if not csv_path.exists():
        return False
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        return len(df) >= budget
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    cfg = load_pilot_config(args.config if args.config.is_absolute() else REPO_ROOT / args.config)
    sb = cfg["sim_budget"]
    targets = cfg["targets"]
    target = float(targets["split_ratio_1550"])
    tol = float(sb.get("tolerance", targets.get("split_ratio_tolerance", 0.05)))
    budgets = sb["budgets_pilot"] if args.pilot else sb["budgets"]
    if args.budgets is not None:
        budgets = args.budgets
    if args.budget is not None:
        budgets = [args.budget]

    meep_cfg = dict(cfg["meep"])
    recipe = MeepRecipe.for_version(meep_cfg["recipe_version"], meep_cfg)
    require_meep()

    from nano_inv.champions import load_champion_centers

    ref = np.load(REPO_ROOT / "data/phase0/latents/ref_published_latent.npy").astype(np.float32)
    champ_cfg = cfg.get("champions", {})
    centers = load_champion_centers(
        REPO_ROOT, champ_cfg.get("latent_paths") if champ_cfg.get("enabled", True) else None
    )
    diverse_top_k = bool(sb.get("diverse_top_k", False))
    diverse_min_h = float(sb.get("diverse_min_hamming", 0.02))
    search_cfg = SearchConfig(
        target_split_ratio=target,
        tolerance=tol,
        objective=cfg.get("search", {}).get("objective", "multi"),
        max_insertion_loss_db=float(targets.get("max_insertion_loss_db", 12.0)),
    )
    s_min, s_max = sb["sigma_range"]
    out_root = REPO_ROOT / sb["output_dir"]
    rep_cfg = cfg.get("replication", {})
    if args.replicate_id is not None:
        rep_root = rep_cfg.get("output_dir", "data/phase1/wedge_a/sim_budget/replicates")
        out_root = REPO_ROOT / rep_root / f"run_{args.replicate_id:02d}"
        out_root.mkdir(parents=True, exist_ok=True)

    policies = sb["policies"] if args.policy == "all" else [args.policy]
    sur_path = REPO_ROOT / cfg["surrogate"]["output_dir"]
    cand_pool = out_root / "candidates_pool.csv"
    mult = int(sb.get("proposals_multiplier", 25))
    default_frac = float(sb.get("hierarchical_sigma_fraction", 0.5))
    h_span = float(sb.get("hierarchical_sigma_span", 0.012))
    h_span_wide = float(sb.get("hierarchical_sigma_span_wide", 0.02))

    if args.ensure_candidates or not cand_pool.exists():
        subprocess.run(
            [
                str(VENV_PY),
                "scripts/generate_ranked_candidates.py",
                "--config",
                str(args.config),
                "--surrogate",
                str(sur_path.relative_to(REPO_ROOT)),
                "--n-proposals",
                str(max(budgets) * mult),
                "--output",
                str(cand_pool.relative_to(REPO_ROOT)),
                "--seed",
                str(args.seed),
                "--latent-only",
            ],
            cwd=REPO_ROOT,
            check=True,
        )

    report: dict = {
        "budgets": budgets,
        "policies": {},
        "target": target,
        "tolerance": tol,
        "seed": args.seed,
        "replicate_id": args.replicate_id,
        "surrogate_dir": str(sur_path.relative_to(REPO_ROOT)),
    }
    rng = np.random.default_rng(args.seed)

    for policy in policies:
        report["policies"][policy] = {}
        for budget in budgets:
            run_dir = out_root / f"{policy}_B{budget}"
            run_dir.mkdir(parents=True, exist_ok=True)
            if args.only_missing and run_is_complete(run_dir, budget):
                print(f"skip complete: {policy} B={budget}")
                import pandas as pd

                prev = pd.read_csv(run_dir / "meep_results.csv")
                rows = []
                for _, r in prev.iterrows():
                    rows.append(
                        MeepRow(
                            sample_id=str(r["sample_id"]),
                            split_ratio_upper=float(r["split_ratio_upper"]),
                            insertion_loss_db=float(r["insertion_loss_db"])
                            if pd.notna(r.get("insertion_loss_db"))
                            else float("nan"),
                            in_spec=bool(r.get("in_spec", False)),
                            sigma=float(r["sigma"]) if pd.notna(r.get("sigma")) else None,
                            policy=str(r.get("policy", policy)),
                            meep_budget=int(r.get("meep_budget", budget)),
                        )
                    )
            elif policy == "random_perturb":
                rows = policy_random_perturb(
                    budget,
                    ref=ref,
                    recipe=recipe,
                    target=target,
                    tol=tol,
                    search_cfg=search_cfg,
                    rng=rng,
                    s_min=s_min,
                    s_max=s_max,
                    out_dir=run_dir,
                    centers=centers,
                )
            elif policy == "sigma_meep":
                rows = policy_sigma_meep(
                    budget, ref=ref, recipe=recipe, target=target, tol=tol,
                    search_cfg=search_cfg, seed=args.seed + budget, s_min=s_min, s_max=s_max, out_dir=run_dir,
                )
            elif policy == "surrogate_rank":
                rows = policy_surrogate_rank(
                    budget,
                    candidates_csv=cand_pool,
                    recipe=recipe,
                    target=target,
                    tol=tol,
                    search_cfg=search_cfg,
                    out_dir=run_dir,
                    diverse_top_k=diverse_top_k,
                    diverse_min_hamming=diverse_min_h,
                )
            elif policy == "hierarchical_wide":
                rows = policy_hierarchical(
                    budget,
                    ref=ref,
                    recipe=recipe,
                    target=target,
                    tol=tol,
                    search_cfg=search_cfg,
                    seed=args.seed + budget,
                    s_min=s_min,
                    s_max=s_max,
                    cfg_path=args.config,
                    sur_path=sur_path,
                    out_dir=run_dir,
                    mult=mult,
                    frac=default_frac,
                    sigma_span=h_span_wide,
                    policy_label=policy,
                    diverse_top_k=diverse_top_k,
                    diverse_min_hamming=diverse_min_h,
                )
            elif policy.startswith("hierarchical"):
                frac = parse_hierarchical_frac(policy, default_frac)
                rows = policy_hierarchical(
                    budget,
                    ref=ref,
                    recipe=recipe,
                    target=target,
                    tol=tol,
                    search_cfg=search_cfg,
                    seed=args.seed + budget,
                    s_min=s_min,
                    s_max=s_max,
                    cfg_path=args.config,
                    sur_path=sur_path,
                    out_dir=run_dir,
                    mult=mult,
                    frac=frac,
                    sigma_span=h_span,
                    policy_label=policy,
                    diverse_top_k=diverse_top_k,
                    diverse_min_hamming=diverse_min_h,
                )
            else:
                raise SystemExit(f"unknown policy {policy}")

            df = rows_to_dataframe(rows)
            df.to_csv(run_dir / "meep_results.csv", index=False)
            summary = summarize_meep_rows(
                rows, policy=policy, meep_budget=budget, target=target, tol=tol, topk=min(20, budget),
            )
            report["policies"][policy][str(budget)] = summary.to_dict()
            print(json.dumps(summary.to_dict(), indent=2))

    if args.metrics_out:
        metrics_path = args.metrics_out if args.metrics_out.is_absolute() else REPO_ROOT / args.metrics_out
    elif args.replicate_id is not None:
        metrics_path = out_root / "run_report.json"
        agg = REPO_ROOT / rep_cfg.get(
            "metrics_aggregate", "data/phase1/wedge_a/sim_budget_replicates.json"
        )
        agg_path = agg if Path(agg).is_absolute() else REPO_ROOT / agg
        if agg_path.exists():
            prev = json.loads(agg_path.read_text())
        else:
            prev = {"runs": []}
        prev.setdefault("runs", []).append(report)
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        agg_path.write_text(json.dumps(prev, indent=2))
    else:
        metrics_path = REPO_ROOT / cfg["data"]["wedge_root"] / "wedge_a_metrics.json"

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if args.replicate_id is not None:
        metrics_path.write_text(json.dumps(report, indent=2))
    elif metrics_path.exists() and not args.replace_run:
        prev = json.loads(metrics_path.read_text())
        prev.setdefault("runs", []).append(report)
        metrics_path.write_text(json.dumps(prev, indent=2))
    else:
        metrics_path.write_text(json.dumps({"runs": [report]}, indent=2))
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
