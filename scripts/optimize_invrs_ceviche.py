#!/usr/bin/env python3
"""Topology optimization on invrs-gym Ceviche challenge (Track C, JAX-only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import invrs_gym.challenges as gym_challenges
import invrs_opt
from invrs_opt.optimizers.lbfgsb import is_converged as lbfgsb_converged

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data/phase1/invrs_benchmark"
DEFAULT_CFG = REPO / "configs" / "invrs_ceviche.yaml"


def get_challenge(name: str):
    if not hasattr(gym_challenges, name):
        raise ValueError(f"Unknown challenge {name!r}")
    return getattr(gym_challenges, name)()


def run_opt(
    challenge_name: str,
    steps: int,
    seed: int,
    beta: float,
    *,
    patience: int = 5,
    min_delta: float = 1e-4,
) -> dict:
    challenge = get_challenge(challenge_name)

    def loss_fn(params):
        response, aux = challenge.component.response(params)
        loss = challenge.loss(response)
        em = challenge.eval_metric(response)
        metrics = challenge.metrics(response, params, aux)
        return loss, (response, em, metrics, aux)

    value_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    opt = invrs_opt.density_lbfgsb(beta=beta)
    params = challenge.component.init(jax.random.PRNGKey(seed))
    state = opt.init(params)

    history: list[dict] = []
    init_loss, (_, init_em, _, _) = loss_fn(params)
    history.append({"step": 0, "loss": float(init_loss), "eval_metric": float(init_em)})

    best_em = float("inf")
    best_step = 0
    stale_checks = 0
    log_every = max(1, steps // 10)
    stopped_reason = "max_steps"

    def note_eval(step: int, em: float) -> None:
        nonlocal best_em, best_step, stale_checks, stopped_reason
        if em >= 0.0 and em < best_em - min_delta:
            best_em = em
            best_step = step
            stale_checks = 0
        elif best_em < float("inf"):
            stale_checks += 1
            if patience > 0 and stale_checks >= patience:
                stopped_reason = (
                    f"patience ({patience} checks without improving in-spec eval_metric)"
                )

    if float(init_em) >= 0.0:
        note_eval(0, float(init_em))

    for i in range(1, steps + 1):
        params = opt.params(state)
        (value, (_, eval_metric, metrics, _)), grad = value_and_grad_fn(params)
        state = opt.update(grad=grad, value=value, params=params, state=state)
        em = float(eval_metric)

        if i % log_every == 0 or i == steps or lbfgsb_converged(state):
            history.append(
                {
                    "step": i,
                    "loss": float(value),
                    "eval_metric": em,
                    "binarization_degree": float(
                        metrics.get("binarization_degree", float("nan"))
                    ),
                    "converged": bool(lbfgsb_converged(state)),
                }
            )
            note_eval(i, em)
            if stopped_reason.startswith("patience"):
                break

        if lbfgsb_converged(state):
            stopped_reason = "lbfgsb_converged"
            break

    final_params = opt.params(state)
    _, (_, final_em, final_metrics, _) = loss_fn(final_params)
    return {
        "challenge": challenge_name,
        "seed": seed,
        "beta": beta,
        "patience": patience,
        "min_delta": min_delta,
        "steps_requested": steps,
        "steps_run": history[-1]["step"],
        "converged": bool(lbfgsb_converged(state)),
        "stopped_reason": stopped_reason,
        "final_loss": history[-1]["loss"],
        "final_eval_metric": float(final_em),
        "best_eval_metric": best_em if best_em < float("inf") else float(final_em),
        "best_step": best_step,
        "in_spec": float(final_em) >= 0.0,
        "best_in_spec": best_em < float("inf"),
        "final_binarization_degree": float(
            final_metrics.get("binarization_degree", float("nan"))
        ),
        "history": history,
    }


def main() -> None:
    import yaml

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--challenge", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Stop after N logged checks without improving eval_metric (0=off)",
    )
    p.add_argument("--min-delta", type=float, default=None, help="Min eval_metric improvement")
    p.add_argument(
        "--out-name",
        default=None,
        help="Output JSON basename (default: {challenge}_opt.json)",
    )
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text()) if args.config.exists() else {}
    opt_cfg = cfg.get("optimizer") or {}
    name = args.challenge or cfg.get("challenge", "ceviche_lightweight_power_splitter")
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))
    steps = args.steps if args.steps is not None else int(opt_cfg.get("steps", 200))
    beta = args.beta if args.beta is not None else float(opt_cfg.get("beta", 4.0))
    patience = (
        args.patience if args.patience is not None else int(opt_cfg.get("patience", 5))
    )
    min_delta = (
        args.min_delta
        if args.min_delta is not None
        else float(opt_cfg.get("min_delta", 1e-4))
    )

    result = run_opt(
        name, steps, seed, beta, patience=patience, min_delta=min_delta
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out_name = args.out_name or f"{name}_opt.json"
    out_path = OUT / out_name
    out_path.write_text(json.dumps(result, indent=2))

    print(f"challenge: {name}")
    print(
        f"best eval_metric: {result['best_eval_metric']:.4f} @ step {result['best_step']}  "
        f"final: {result['final_eval_metric']:.4f}  in_spec: {result['in_spec']}"
    )
    print(
        f"steps: {result['steps_run']}/{steps}  converged: {result['converged']}  "
        f"stopped: {result['stopped_reason']}"
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
