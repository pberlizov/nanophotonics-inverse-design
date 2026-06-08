#!/usr/bin/env python3
"""Refine champion latent z on invrs-gym Ceviche via grad through decode_soft (Path A).

  z → drcgenerator G(z) [soft] → Ceviche density → gym loss → ∂loss/∂z

Does not use MEEP (run MEEP verify separately). Compare to gym-native opt in
`optimize_invrs_ceviche.py` (optimizes density directly, no manifold).

  PYTHONPATH=src python scripts/refine_champion_grad.py
  PYTHONPATH=src python scripts/refine_champion_grad.py --sample-id local_00022 --steps 50
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nano_inv.invrs_adapter import (
    ChallengeName,
    gym_loss_and_metric,
    latent_to_gym_params,
    template_density,
)
from nano_inv.latent import pad_latent_to_standard
from nano_inv.manifold import EBeamManifold

DEFAULT_CFG = REPO / "configs" / "refine_champion_grad.yaml"
OUT = REPO / "data" / "phase1" / "invrs_benchmark"


def load_cfg(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def refine_latent(
    z0: jnp.ndarray,
    *,
    challenge: ChallengeName,
    template_seed: int,
    steps: int,
    learning_rate: float,
    grad_clip_norm: float,
    manifold: EBeamManifold,
    patience: int = 8,
    min_delta: float = 1e-4,
) -> dict:
    template = template_density(challenge, seed=template_seed)
    z = jnp.asarray(z0)

    def loss_fn(latent: jnp.ndarray) -> jnp.ndarray:
        loss, _em = gym_loss_and_metric(latent, template, manifold, challenge)
        return loss

    loss_and_grad = jax.value_and_grad(loss_fn)
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(learning_rate),
    )
    opt_state = optimizer.init(z)

    history: list[dict] = []
    loss0, em0 = gym_loss_and_metric(z, template, manifold, challenge)
    em0_f = float(em0)
    history.append(
        {
            "step": 0,
            "loss": float(loss0),
            "eval_metric": em0_f,
            "grad_norm": 0.0,
        }
    )

    best_em = em0_f if em0_f >= 0.0 else float("inf")
    best_step = 0 if em0_f >= 0.0 else 0
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

    for step in range(1, steps + 1):
        loss, grad = loss_and_grad(z)
        updates, opt_state = optimizer.update(grad, opt_state)
        z = optax.apply_updates(z, updates)
        z = jnp.clip(z, manifold.model.eps, 1.0)
        # Track in-spec best every step; sparse history for JSON size.
        _, em = gym_loss_and_metric(z, template, manifold, challenge)
        em_f = float(em)
        note_eval(step, em_f)
        if step % log_every == 0 or step == steps:
            gnorm = float(jnp.linalg.norm(grad))
            history.append(
                {
                    "step": step,
                    "loss": float(loss),
                    "eval_metric": em_f,
                    "grad_norm": gnorm,
                }
            )
            if stopped_reason.startswith("patience"):
                break

    loss_f, em_f = gym_loss_and_metric(z, template, manifold, challenge)
    em_f = float(em_f)
    return {
        "final_z": np.asarray(z),
        "final_loss": float(loss_f),
        "final_eval_metric": em_f,
        "best_eval_metric": best_em if best_em < float("inf") else em_f,
        "best_step": best_step,
        "in_spec": em_f >= 0.0,
        "best_in_spec": best_em < float("inf"),
        "stopped_reason": stopped_reason,
        "history": history,
        "delta_eval_metric": float(em_f - em0_f),
        "delta_best_eval_metric": float(best_em - em0_f),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--sample-id", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    args = p.parse_args()

    cfg = load_cfg(args.config)
    challenge: ChallengeName = cfg.get("challenge", "ceviche_lightweight_power_splitter")
    opt_cfg = cfg.get("optimizer") or {}
    steps = args.steps if args.steps is not None else int(opt_cfg.get("steps", 40))
    lr = args.learning_rate if args.learning_rate is not None else float(
        opt_cfg.get("learning_rate", 0.02)
    )
    grad_clip = float(opt_cfg.get("grad_clip_norm", 1.0))
    patience = int(opt_cfg.get("patience", 8))
    min_delta = float(opt_cfg.get("min_delta", 1e-4))
    template_seed = int(cfg.get("template_seed", 0))

    champions = cfg.get("champions") or []
    if args.sample_id:
        champions = [c for c in champions if c["id"] == args.sample_id]

    manifold = EBeamManifold.load()
    rows: list[dict] = []

    for ch in champions:
        sid = ch["id"]
        latent_path = REPO / ch["latent"]
        if not latent_path.exists():
            print(f"SKIP {sid}: missing {latent_path}")
            continue
        z0 = jnp.asarray(pad_latent_to_standard(np.load(latent_path)))
        print(f"==> refine {sid} ({steps} steps, lr={lr})")
        result = refine_latent(
            z0,
            challenge=challenge,
            template_seed=template_seed,
            steps=steps,
            learning_rate=lr,
            grad_clip_norm=grad_clip,
            manifold=manifold,
            patience=patience,
            min_delta=min_delta,
        )
        row = {
            "sample_id": sid,
            "latent_path": str(latent_path.relative_to(REPO)),
            "challenge": challenge,
            "steps": steps,
            "learning_rate": lr,
            **{k: v for k, v in result.items() if k != "final_z"},
        }
        rows.append(row)
        print(
            f"    eval_metric {row['history'][0]['eval_metric']:.4f} → "
            f"best {result['best_eval_metric']:.4f} @ {result['best_step']}  "
            f"final {result['final_eval_metric']:.4f}  "
            f"({result['stopped_reason']})  in_spec={result['in_spec']}"
        )
        out_z = OUT / f"{sid}_refined_latent.npy"
        np.save(out_z, result["final_z"])
        row["refined_latent_path"] = str(out_z.relative_to(REPO))

    OUT.mkdir(parents=True, exist_ok=True)
    out_json = OUT / "refine_champion_grad.json"
    merged: list[dict] = []
    if out_json.exists() and args.sample_id:
        for old in json.loads(out_json.read_text()):
            if old.get("sample_id") != args.sample_id:
                merged.append(old)
    merged.extend(rows)
    out_json.write_text(json.dumps(merged, indent=2))
    rows = merged

    lines = [
        "# Champion grad refinement on invrs-gym Ceviche",
        "",
        f"Challenge: `{challenge}` · optimizer: Adam, {steps} steps, lr={lr}",
        "",
        "Compare to gym-native density opt: `ceviche_lightweight_power_splitter_opt.json` (~0.085 eval_metric).",
        "",
        "| sample | start → best (step) → final | in_spec |",
        "|--------|-------------------------|---------|",
    ]
    for r in rows:
        s0 = r["history"][0]["eval_metric"]
        sb = r.get("best_eval_metric", r["final_eval_metric"])
        s1 = r["final_eval_metric"]
        flag = "✓" if r.get("best_in_spec", r["in_spec"]) else "—"
        lines.append(
            f"| {r['sample_id']} | {s0:.4f} → **{sb:.4f}** ({r.get('best_step', '?')}) "
            f"→ {s1:.4f} | {flag} |"
        )

    out_md = OUT / "refine_champion_grad.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_json}\nWrote {out_md}")


if __name__ == "__main__":
    main()
