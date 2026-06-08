#!/usr/bin/env python3
"""Evaluate invrs-gym Ceviche challenges (external SOTA benchmark track).

This does NOT use our MEEP template or drcgenerator masks yet — it establishes
a baseline on the standardized gym API before wiring Path A (z -> G(z) -> diff EM).

Install: uv pip install invrs-gym --python .venv/bin/python
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO / "configs" / "invrs_ceviche.yaml"
OUT = REPO / "data" / "phase1" / "invrs_benchmark"

CHALLENGE_NAMES = (
    "ceviche_lightweight_power_splitter",
    "ceviche_power_splitter",
    "ceviche_lightweight_beam_splitter",
    "ceviche_beam_splitter",
)


def load_cfg(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def get_challenge(name: str):
    import invrs_gym.challenges as ch

    if not hasattr(ch, name):
        raise ValueError(f"Unknown challenge {name!r}; expected one of {CHALLENGE_NAMES}")
    return getattr(ch, name)()


def eval_params(challenge, params) -> dict:
    response, aux = challenge.component.response(params)
    loss = challenge.loss(response)
    eval_metric = challenge.eval_metric(response)
    metrics = challenge.metrics(response, params, aux)
    transmission = jnp.abs(response.s_parameters) ** 2
    return {
        "loss": float(loss),
        "eval_metric": float(eval_metric),
        "in_spec": float(eval_metric) >= 0.0,
        "transmission_mean": float(jnp.mean(transmission)),
        "transmission_min": float(jnp.min(transmission)),
        "transmission_max": float(jnp.max(transmission)),
        "metrics": {k: float(v) for k, v in metrics.items() if hasattr(v, "item")},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="invrs-gym Ceviche benchmark spike")
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--challenge", default=None, help="Override config challenge name")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    cfg = load_cfg(args.config)
    name = args.challenge or cfg["challenge"]
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 0))

    challenge = get_challenge(name)
    key = jax.random.PRNGKey(seed)
    params = challenge.component.init(key)
    result = eval_params(challenge, params)
    result.update(
        {
            "challenge": name,
            "seed": seed,
            "note": (
                "eval_metric >= 0 means gym target window hit; compare to invrs-leaderboard "
                "entries for the same challenge. Random init is expected to be negative."
            ),
        }
    )

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{name}_baseline.json"
    out_path.write_text(json.dumps(result, indent=2))

    print(f"challenge: {name}")
    print(f"eval_metric: {result['eval_metric']:.4f}  in_spec: {result['in_spec']}")
    print(f"transmission mean/min/max: {result['transmission_mean']:.4f} / "
          f"{result['transmission_min']:.4f} / {result['transmission_max']:.4f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
