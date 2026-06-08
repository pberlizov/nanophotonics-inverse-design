#!/usr/bin/env python3
"""Assemble pilot deliverable dossier: champions, sim-budget tops, PNG/GDS, design cards."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nano_inv.drc_heuristic import check_mask_heuristic  # noqa: E402
from nano_inv.pilot import (  # noqa: E402
    REPO_ROOT as ROOT,
    as_repo_relative,
    build_template_context,
    deliverables_dir,
    load_pilot_config,
    render_template,
    resolve_path,
)
from nano_inv.sim_budget import is_in_spec  # noqa: E402

VENV_PY = REPO_ROOT / ".venv/bin/python"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs/pilot/benchmark_50_50.yaml")
    return p.parse_args()


def export_png(mask_path: Path, sample_id: str, out_png: Path, title_extra: str = "") -> bool:
    try:
        import matplotlib.pyplot as plt

        mask = np.load(mask_path)
        plt.figure(figsize=(5, 5))
        plt.imshow(mask, cmap="gray", interpolation="nearest")
        plt.title(f"{sample_id}{title_extra}")
        plt.axis("off")
        plt.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200)
        plt.close()
        return True
    except ImportError:
        return False


def export_gds(mask_path: Path, sample_id: str, out_dir: Path, pitch_um: float) -> Path | None:
    gds_script = REPO_ROOT / "scripts/export_layout_gds.py"
    try:
        subprocess.run(
            [
                str(VENV_PY),
                str(gds_script),
                "--mask",
                str(mask_path.relative_to(REPO_ROOT)),
                "--sample-id",
                sample_id,
                "--output-dir",
                str(out_dir.relative_to(REPO_ROOT)),
                "--pitch-um",
                str(pitch_um),
            ],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
        )
        gds = out_dir / f"{sample_id}.gds"
        return gds if gds.exists() else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def collect_sim_budget_designs(cfg: dict) -> list[dict]:
    src = cfg.get("sources", {})
    policy = src.get("sim_budget_policy", "surrogate_rank")
    budget = int(src.get("sim_budget_budget", 30))
    sb_dir = resolve_path(src.get("sim_budget_dir", "data/phase1/wedge_a/sim_budget"))
    meep_csv = sb_dir / f"{policy}_B{budget}" / "meep_results.csv"
    if not meep_csv.exists():
        return []

    targets = cfg["targets"]
    target = float(targets["split_ratio_1550"])
    tol = float(cfg.get("sim_budget", {}).get("tolerance", targets.get("split_ratio_tolerance", 0.05)))

    df = pd.read_csv(meep_csv)
    df = df[np.isfinite(df["split_ratio_upper"])].copy()
    df["abs_err"] = (df["split_ratio_upper"] - target).abs()
    df = df.sort_values("abs_err")
    top_k = int(cfg.get("deliverables", {}).get("top_k", 5))
    rows: list[dict] = []

    pool_csv = sb_dir / "candidates_pool.csv"
    pool = pd.read_csv(pool_csv) if pool_csv.exists() else None

    for _, r in df.head(top_k).iterrows():
        sid = str(r["sample_id"])
        mask_path = None
        if pool is not None and "sample_id" in pool.columns:
            m = pool[pool["sample_id"].astype(str) == sid]
            if len(m):
                mask_path = resolve_path(m.iloc[0]["mask_path"])
        if mask_path is None or not mask_path.exists():
            for cand in (
                sb_dir / f"{policy}_B{budget}" / "candidates/masks",
                sb_dir / "masks",
            ):
                p = cand / f"{sid}_mask.npy"
                if p.exists():
                    mask_path = p
                    break
        if mask_path is None or not mask_path.exists():
            continue
        rows.append(
            {
                "sample_id": sid,
                "mask_path": as_repo_relative(mask_path),
                "split_ratio_upper": float(r["split_ratio_upper"]),
                "insertion_loss_db": float(r["insertion_loss_db"]) if pd.notna(r.get("insertion_loss_db")) else None,
                "in_spec": bool(r.get("in_spec", is_in_spec(float(r["split_ratio_upper"]), target, tol))),
                "source": f"sim_budget/{policy}_B{budget}",
                "policy": policy,
                "meep_budget": budget,
            }
        )
    return rows


def design_card(
    entry: dict,
    *,
    target: float,
    tol: float,
    mask_path: Path,
    dlv_dir: Path,
    include_gds: bool,
    pitch_um: float,
) -> dict:
    sid = entry["sample_id"]
    design_dir = dlv_dir / "designs" / sid
    design_dir.mkdir(parents=True, exist_ok=True)

    mask = np.load(mask_path)
    drc = check_mask_heuristic(mask)
    split = entry.get("split_ratio_upper")
    card = {
        "sample_id": sid,
        "mask_path": entry.get("mask_path", as_repo_relative(mask_path)),
        "split_ratio_upper": split,
        "insertion_loss_db": entry.get("insertion_loss_db"),
        "in_spec": entry.get("in_spec", is_in_spec(float(split), target, tol) if split is not None else False),
        "target_split_ratio": target,
        "tolerance": tol,
        "source": entry.get("source", ""),
        "drc_heuristic_passed": drc.passed,
        "drc_fill_ratio": drc.fill_ratio,
        "drc_min_run_length": drc.min_run_length,
        "drc_reasons": drc.reasons,
    }

    shutil.copy2(mask_path, design_dir / f"{sid}_mask.npy")
    png_ok = export_png(
        mask_path,
        sid,
        design_dir / f"{sid}_mask.png",
        title_extra=f"\nsplit={split:.3f}" if split is not None else "",
    )
    card["png_exported"] = png_ok

    gds_path = None
    if include_gds:
        gds_path = export_gds(mask_path, sid, design_dir, pitch_um)
    card["gds_path"] = as_repo_relative(gds_path) if gds_path else None

    (design_dir / "design_card.json").write_text(json.dumps(card, indent=2))
    return card


def main() -> None:
    args = parse_args()
    cfg = load_pilot_config(args.config)
    out = deliverables_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)

    targets = cfg["targets"]
    target = float(targets["split_ratio_1550"])
    tol = float(cfg.get("sim_budget", {}).get("tolerance", targets.get("split_ratio_tolerance", 0.05)))
    dlv = cfg.get("deliverables", {})
    include_gds = bool(dlv.get("include_gds", True))
    pitch_um = float(dlv.get("pitch_um", 4.0))

    seen: set[str] = set()
    all_cards: list[dict] = []

    for ch in cfg.get("champions") or []:
        sid = str(ch["sample_id"])
        if sid in seen:
            continue
        seen.add(sid)
        mask_path = resolve_path(ch["mask_path"])
        if not mask_path.exists():
            print(f"skip missing champion mask: {mask_path}")
            continue
        entry = dict(ch)
        entry.setdefault("source", ch.get("source", "champion"))
        card = design_card(
            entry,
            target=target,
            tol=tol,
            mask_path=mask_path,
            dlv_dir=out,
            include_gds=include_gds,
            pitch_um=pitch_um,
        )
        all_cards.append(card)

    for entry in collect_sim_budget_designs(cfg):
        sid = entry["sample_id"]
        if sid in seen:
            continue
        seen.add(sid)
        mask_path = resolve_path(entry["mask_path"])
        card = design_card(
            entry,
            target=target,
            tol=tol,
            mask_path=mask_path,
            dlv_dir=out,
            include_gds=include_gds,
            pitch_um=pitch_um,
        )
        all_cards.append(card)

    dossier = pd.DataFrame(all_cards)
    dossier_path = out / "design_dossier.csv"
    dossier.to_csv(dossier_path, index=False)

    # Sim contract from template
    tpl = REPO_ROOT / "templates/pilot/sim_contract.md"
    if tpl.exists():
        ctx = build_template_context(cfg)
        contract = render_template(tpl, ctx)
        (out / "SIM_CONTRACT.md").write_text(contract)
        shutil.copy2(tpl, out / "SIM_CONTRACT_TEMPLATE.md")

    manifest = {
        "pilot_id": cfg.get("pilot", {}).get("id"),
        "n_designs": len(all_cards),
        "dossier_csv": as_repo_relative(dossier_path),
        "target_split_ratio": target,
        "tolerance": tol,
        "recipe_version": cfg.get("meep", {}).get("recipe_version"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
