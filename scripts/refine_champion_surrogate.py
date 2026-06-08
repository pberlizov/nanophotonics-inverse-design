#!/usr/bin/env python3
"""Refine champion latent z on MEEP-aligned surrogate (hard/STE decode for mask_mlp).

  z → decode_{ste|hard|soft} → mask features → mask_mlp pred → (pred - 0.5)² + trust region

Verify with `verify_refined_champions.py --refine-source surrogate` + MEEP prod r25.

  PYTHONPATH=src python scripts/refine_champion_surrogate.py --sample-id local_00022 --steps 40
  PYTHONPATH=src python scripts/refine_champion_product.py --sample-id local_00022
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nano_inv.latent import pad_latent_to_standard
from nano_inv.manifold import EBeamManifold
from nano_inv.surrogate import MASK_SHAPE, SurrogateArtifact, load_artifact

DEFAULT_CFG = REPO / "configs" / "refine_champion_surrogate.yaml"
SURROGATE_IMPROVED = REPO / "data/phase1/wedge_a/surrogate_improved/surrogate.joblib"
DEFAULT_CORPUS_SIM = (
    REPO / "data/phase1/sim_results_phase0_v1_all.csv",
    REPO / "data/phase0/sim_results_phase0_v1_all.csv",
)
TRAINING_ROWS_FALLBACK = REPO / "data/phase1/wedge_a/surrogate_improved/training_rows.csv"
DecodeMode = Literal["soft", "hard", "ste"]


def load_corpus_prod_split(
    sample_id: str,
    *,
    corpus_paths: tuple[Path, ...] = DEFAULT_CORPUS_SIM,
    training_rows: Path = TRAINING_ROWS_FALLBACK,
) -> float | None:
    """MEEP prod label: phase0_v1 @ resolution 25 from corpus or training_rows."""
    for path in corpus_paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "sample_id" not in df.columns:
            continue
        sub = df[df["sample_id"] == sample_id]
        if "recipe_version" in sub.columns:
            sub = sub[sub["recipe_version"].fillna("phase0_v1") == "phase0_v1"]
        if "resolution" in sub.columns:
            sub = sub[sub["resolution"].fillna(25) == 25]
        if "status" in sub.columns:
            sub = sub[sub["status"].fillna("ok") == "ok"]
        if len(sub) and "split_ratio_upper" in sub.columns:
            val = sub.iloc[0]["split_ratio_upper"]
            if pd.notna(val):
                return float(val)
    if training_rows.exists():
        df = pd.read_csv(training_rows)
        sub = df[df["sample_id"] == sample_id]
        if len(sub) and "split_ratio_upper" in sub.columns:
            val = sub.iloc[0]["split_ratio_upper"]
            if pd.notna(val):
                return float(val)
    return None


def corpus_on_target(
    sample_id: str,
    *,
    split_tol: float = 0.03,
    target: float = 0.5,
    **kwargs,
) -> tuple[bool, float | None]:
    prod = load_corpus_prod_split(sample_id, **kwargs)
    if prod is None:
        return False, None
    return abs(prod - target) < split_tol, prod


def load_cfg(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_surrogate_path(cfg: dict) -> Path:
    if cfg.get("prefer_surrogate_improved", True) and SURROGATE_IMPROVED.exists():
        return SURROGATE_IMPROVED
    raw = Path(cfg["surrogate"])
    path = raw if raw.is_absolute() else REPO / raw
    if path.is_dir():
        return path / "surrogate.joblib"
    if path.suffix != ".joblib" and (path / "surrogate.joblib").exists():
        return path / "surrogate.joblib"
    return path


def normalize_mask_jax(mask: jnp.ndarray) -> jnp.ndarray:
    """Match ``surrogate.normalize_mask_to_standard`` for pooled features."""
    m = jnp.squeeze(mask).astype(jnp.float32)
    h, w = m.shape
    th, tw = MASK_SHAPE
    if (h, w) == (th, tw):
        return m
    if h > th or w > tw:
        ih = (h - th) // 2
        iw = (w - tw) // 2
        return m[ih : ih + th, iw : iw + tw]
    out = jnp.zeros((th, tw), dtype=jnp.float32)
    dh = (th - h) // 2
    dw = (tw - w) // 2
    return out.at[dh : dh + h, dw : dw + w].set(m)


def pool_mask_jax(mask: jnp.ndarray, pool: int) -> jnp.ndarray:
    m = normalize_mask_jax(mask)
    h, w = m.shape
    h2, w2 = h // pool * pool, w // pool * pool
    m = mask[:h2, :w2]
    return jnp.max(m.reshape(h2 // pool, pool, w2 // pool, pool), axis=(1, 3))


def sklearn_mlp_forward(x: jnp.ndarray, pipeline) -> jnp.ndarray:
    scaler = pipeline.named_steps["scale"]
    mlp = pipeline.named_steps["mlp"]
    mean = jnp.asarray(scaler.mean_, dtype=jnp.float32)
    scale = jnp.asarray(scaler.scale_, dtype=jnp.float32)
    x = (x - mean) / scale
    for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        x = x @ jnp.asarray(w, dtype=jnp.float32) + jnp.asarray(b, dtype=jnp.float32)
        if i < len(mlp.coefs_) - 1:
            x = jnp.maximum(x, 0.0)
    return x.squeeze()


def decode_mask_jax(
    manifold: EBeamManifold,
    latent: jnp.ndarray,
    mode: DecodeMode,
) -> jnp.ndarray:
    if mode == "soft":
        return jnp.squeeze(manifold.decode_soft(latent))
    if mode == "ste":
        return jnp.squeeze(manifold.decode_ste(latent))
    hard = manifold.decode(latent)
    return jax.lax.stop_gradient(jnp.squeeze(hard))


def make_jax_predict(
    artifact: SurrogateArtifact,
    *,
    manifold: EBeamManifold,
    sigma: float,
    decode_mode: DecodeMode = "ste",
):
    arch = artifact.architecture
    pipe = artifact.pipeline

    if arch == "latent_mlp":

        def predict(latent: jnp.ndarray) -> jnp.ndarray:
            flat = latent.reshape(-1).astype(jnp.float32)
            return sklearn_mlp_forward(flat, pipe)

        return predict

    if arch == "mask_mlp":
        pool = int(artifact.mask_pool)
        sig = float(sigma)

        def predict(latent: jnp.ndarray) -> jnp.ndarray:
            rho = decode_mask_jax(manifold, latent, decode_mode)
            feats = pool_mask_jax(rho, pool).reshape(-1)  # normalize inside pool_mask_jax
            if artifact.sigma_feature:
                feats = jnp.concatenate([feats, jnp.array([sig], dtype=jnp.float32)])
            return sklearn_mlp_forward(feats, pipe)

        return predict

    raise ValueError(f"unsupported surrogate architecture {arch!r} for JAX grad refine")


def refine_latent(
    z0: jnp.ndarray,
    *,
    predict_fn,
    target_split: float,
    steps: int,
    learning_rate: float,
    grad_clip_norm: float,
    manifold: EBeamManifold,
    trust_lambda: float = 0.0,
    corpus_prod_split: float | None = None,
    prod_penalty_weight: float = 0.0,
) -> dict:
    z = jnp.asarray(z0)
    z_anchor = jnp.asarray(z0)
    target = float(target_split)

    def loss_fn(latent: jnp.ndarray) -> jnp.ndarray:
        pred = predict_fn(latent)
        loss = (pred - target) ** 2
        if prod_penalty_weight > 0 and corpus_prod_split is not None:
            cp = jnp.asarray(corpus_prod_split, dtype=jnp.float32)
            loss = loss + prod_penalty_weight * (pred - cp) ** 2
        if trust_lambda > 0:
            loss = loss + trust_lambda * jnp.sum((latent - z_anchor) ** 2)
        return loss

    loss_and_grad = jax.value_and_grad(loss_fn)
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(learning_rate),
    )
    opt_state = optimizer.init(z)

    pred0 = float(predict_fn(z))
    loss0 = float((pred0 - target) ** 2)
    history: list[dict] = [
        {
            "step": 0,
            "loss": loss0,
            "pred_split": pred0,
            "abs_err": abs(pred0 - target),
            "grad_norm": 0.0,
            "z_delta_norm": 0.0,
        }
    ]
    best_z = z
    best_loss = loss0
    stall = 0
    early_stop_patience = 5

    for step in range(1, steps + 1):
        loss, grad = loss_and_grad(z)
        loss_f = float(loss)
        if loss_f < best_loss - 1e-8:
            best_loss = loss_f
            best_z = z
            stall = 0
        else:
            stall += 1
        updates, opt_state = optimizer.update(grad, opt_state)
        z = optax.apply_updates(z, updates)
        z = jnp.clip(z, manifold.model.eps, 1.0)
        if step % max(1, steps // 10) == 0 or step == steps:
            pred = float(predict_fn(z))
            history.append(
                {
                    "step": step,
                    "loss": loss_f,
                    "pred_split": pred,
                    "abs_err": abs(pred - target),
                    "grad_norm": float(jnp.linalg.norm(grad)),
                    "z_delta_norm": float(jnp.linalg.norm(z - z_anchor)),
                }
            )
        if stall >= early_stop_patience:
            break

    z = best_z
    pred_f = float(predict_fn(z))
    return {
        "final_z": np.asarray(z),
        "final_loss": float((pred_f - target) ** 2),
        "final_pred_split": pred_f,
        "final_abs_err": abs(pred_f - target),
        "history": history,
        "delta_pred_split": pred_f - pred0,
        "delta_abs_err": abs(pred_f - target) - abs(pred0 - target),
        "final_z_delta_norm": float(np.linalg.norm(np.asarray(z) - np.asarray(z_anchor))),
    }


def resolve_output_dir(cfg: dict, args: argparse.Namespace) -> Path:
    if args.output_dir:
        out = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    else:
        raw = cfg.get("output_dir", "data/phase1/refine_surrogate_ste")
        out = REPO / raw
    suffix = args.run_suffix or cfg.get("run_suffix")
    if suffix:
        out = out.parent / f"{out.name}_{suffix}"
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--sample-id", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument(
        "--decode",
        choices=("soft", "hard", "ste"),
        default=None,
        help="Mask features for mask_mlp (default ste — aligns with hard-trained ranker)",
    )
    p.add_argument("--trust-region-lambda", type=float, default=None)
    p.add_argument("--no-trust-region", action="store_true")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--run-suffix", default=None, help="Append to output dir / JSON basename")
    p.add_argument(
        "--refine-when-off-target-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="No-op at z₀ when corpus prod r25 already |split−0.5| < tol (default: off)",
    )
    p.add_argument("--prod-split-tol", type=float, default=0.03)
    p.add_argument(
        "--prod-penalty-weight",
        type=float,
        default=None,
        help="If >0, add weight×(pred−corpus_prod)² to loss (uses corpus label as target)",
    )
    p.add_argument("--force-refine", action="store_true", help="Refine even when corpus on-target")
    args = p.parse_args()

    cfg = load_cfg(args.config)
    opt_cfg = cfg.get("optimizer") or {}
    trust_cfg = cfg.get("trust_region") or {}
    use_trust = not args.no_trust_region and trust_cfg.get("enabled", True)
    trust_lambda = 0.0
    if use_trust:
        trust_lambda = (
            args.trust_region_lambda
            if args.trust_region_lambda is not None
            else float(trust_cfg.get("lambda", 80.0))
        )

    decode_mode: DecodeMode = (
        args.decode
        or cfg.get("decode", "ste")
    )  # type: ignore[assignment]

    if use_trust:
        steps = args.steps if args.steps is not None else int(
            opt_cfg.get("steps_trust", opt_cfg.get("steps", 40))
        )
        lr = (
            args.learning_rate
            if args.learning_rate is not None
            else float(opt_cfg.get("learning_rate_trust", opt_cfg.get("learning_rate", 0.008)))
        )
    else:
        steps = args.steps if args.steps is not None else int(opt_cfg.get("steps", 50))
        lr = args.learning_rate if args.learning_rate is not None else float(
            opt_cfg.get("learning_rate", 0.02)
        )
    grad_clip = float(opt_cfg.get("grad_clip_norm", 1.0))
    target_split = float(cfg.get("target_split_ratio", cfg.get("target_split", 0.5)))
    default_sigma = float(cfg.get("default_sigma", cfg.get("sigma", 0.02)))
    off_target_only = (
        args.refine_when_off_target_only
        if args.refine_when_off_target_only is not None
        else bool(cfg.get("refine_when_off_target_only", False))
    )
    prod_tol = float(
        args.prod_split_tol if args.prod_split_tol is not None else cfg.get("prod_split_tol", 0.03)
    )
    prod_penalty = (
        args.prod_penalty_weight
        if args.prod_penalty_weight is not None
        else float(cfg.get("prod_penalty_weight", 0.0))
    )

    sur_path = resolve_surrogate_path(cfg)
    if not sur_path.exists():
        raise FileNotFoundError(
            f"Missing {sur_path}. Train: python scripts/train_wedge_a_surrogate.py "
            "--config configs/wedge_a_improved.yaml"
        )
    artifact = load_artifact(sur_path)
    manifold = EBeamManifold.load()

    champions = cfg.get("champions") or []
    if args.sample_id:
        champions = [c for c in champions if c["id"] == args.sample_id]

    out_dir = resolve_output_dir(cfg, args)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_name = "refine_champion_surrogate.json"
    if args.run_suffix or cfg.get("run_suffix"):
        json_name = json_name.replace(".json", f"_{args.run_suffix or cfg.get('run_suffix')}.json")
    rows: list[dict] = []

    for ch in champions:
        sid = ch["id"]
        latent_path = REPO / ch["latent"]
        if not latent_path.exists():
            print(f"SKIP {sid}: missing {latent_path}")
            continue
        sigma = float(ch.get("sigma", default_sigma))
        z0 = jnp.asarray(pad_latent_to_standard(np.load(latent_path)))
        predict_fn = make_jax_predict(
            artifact, manifold=manifold, sigma=sigma, decode_mode=decode_mode
        )

        sk_pred = artifact.predict_from_latent(np.asarray(z0), manifold=manifold, sigma=sigma)
        jax_pred = float(predict_fn(z0))
        on_target, corpus_prod = corpus_on_target(sid, split_tol=prod_tol, target=target_split)
        loss_target = target_split
        if prod_penalty > 0 and corpus_prod is not None:
            loss_target = corpus_prod

        skip_reason = None
        if off_target_only and not args.force_refine and on_target:
            skip_reason = (
                f"corpus prod r25={corpus_prod:.4f} within ±{prod_tol} of {target_split}"
            )

        print(
            f"==> refine {sid} ({steps} steps, lr={lr}, sigma={sigma}, decode={decode_mode}, "
            f"trust_λ={trust_lambda}) [sklearn_hard={sk_pred:.4f} jax={jax_pred:.4f}]"
            + (f" corpus_prod={corpus_prod:.4f}" if corpus_prod is not None else "")
        )
        if skip_reason:
            print(f"    SKIP refine: {skip_reason}")

        if skip_reason:
            pred0 = jax_pred
            result = {
                "final_z": np.asarray(z0),
                "final_loss": float((pred0 - loss_target) ** 2),
                "final_pred_split": pred0,
                "final_abs_err": abs(pred0 - loss_target),
                "history": [
                    {
                        "step": 0,
                        "loss": float((pred0 - loss_target) ** 2),
                        "pred_split": pred0,
                        "abs_err": abs(pred0 - loss_target),
                        "grad_norm": 0.0,
                        "z_delta_norm": 0.0,
                    }
                ],
                "delta_pred_split": 0.0,
                "delta_abs_err": 0.0,
                "final_z_delta_norm": 0.0,
                "refine_skipped": True,
                "refine_skip_reason": skip_reason,
            }
        else:
            result = refine_latent(
                z0,
                predict_fn=predict_fn,
                target_split=loss_target,
                steps=steps,
                learning_rate=lr,
                grad_clip_norm=grad_clip,
                manifold=manifold,
                trust_lambda=trust_lambda,
                corpus_prod_split=corpus_prod,
                prod_penalty_weight=prod_penalty,
            )
            result["refine_skipped"] = False
        row = {
            "sample_id": sid,
            "latent_path": str(latent_path.relative_to(REPO)),
            "surrogate": str(sur_path.parent.relative_to(REPO)),
            "sigma": sigma,
            "target_split_ratio": target_split,
            "steps": steps,
            "learning_rate": lr,
            "decode_mode": decode_mode,
            "trust_region_lambda": trust_lambda,
            "sklearn_pred_start": float(sk_pred),
            "corpus_prod_split": corpus_prod,
            "corpus_on_target": on_target,
            "loss_target_split": loss_target,
            "prod_penalty_weight": prod_penalty,
            **{k: v for k, v in result.items() if k != "final_z"},
        }
        rows.append(row)
        print(
            f"    pred_split {row['history'][0]['pred_split']:.4f} → "
            f"{result['final_pred_split']:.4f}  "
            f"(Δ err {result['delta_abs_err']:+.4f}, ||Δz||={result['final_z_delta_norm']:.4f})"
        )
        out_z = out_dir / f"{sid}_refined_latent.npy"
        np.save(out_z, result["final_z"])
        row["refined_latent_path"] = str(out_z.relative_to(REPO))

    out_json = out_dir / json_name
    out_json.write_text(json.dumps(rows, indent=2))

    md_name = json_name.replace(".json", ".md")
    lines = [
        "# Champion surrogate refinement (MEEP-aligned ranker)",
        "",
        f"Surrogate: `{sur_path.parent.relative_to(REPO)}` · decode=`{decode_mode}` · "
        f"trust λ={trust_lambda} · target split={target_split} · Adam {steps} steps, lr={lr}",
        "",
        "Verify: `verify_refined_champions.py --refine-source surrogate --refine-json ...`",
        "",
        "| sample | pred_split (start → end) | |err| Δ | ||Δz|| |",
        "|--------|--------------------------|-------|--------|",
    ]
    for r in rows:
        s0 = r["history"][0]["pred_split"]
        s1 = r["final_pred_split"]
        lines.append(
            f"| {r['sample_id']} | {s0:.4f} → {s1:.4f} | {r['delta_abs_err']:+.4f} | "
            f"{r['final_z_delta_norm']:.4f} |"
        )

    out_md = out_dir / md_name
    out_md.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {out_json}\nWrote {out_md}")


if __name__ == "__main__":
    main()
