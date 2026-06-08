"""Pilot engagement helpers: config merge, paths, template rendering."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return REPO_ROOT


def resolve_path(path: str | Path, *, root: Path | None = None) -> Path:
    p = Path(path)
    base = root or REPO_ROOT
    return p if p.is_absolute() else (base / p).resolve()


def as_repo_relative(path: Path, *, root: Path | None = None) -> str:
    base = root or REPO_ROOT
    try:
        return str(path.resolve().relative_to(base))
    except ValueError:
        return str(path.resolve())


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, val in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = deepcopy(val)
    return out


def load_pilot_config(config_path: Path) -> dict[str, Any]:
    """Load pilot YAML; recursively merge base_config chain (e.g. validate → production → wedge_a)."""
    cfg_path = resolve_path(config_path)
    overlay = load_yaml(cfg_path)
    base_rel = overlay.get("base_config", "configs/wedge_a.yaml")
    if base_rel is None:
        return overlay
    base_path = resolve_path(base_rel)
    if base_path.resolve() == cfg_path.resolve():
        return overlay
    base = load_pilot_config(base_path)
    merged = deep_merge(base, overlay)
    merged.setdefault("pilot", {})
    merged["pilot"].setdefault("config_path", as_repo_relative(cfg_path))
    return merged


def pilot_id(cfg: dict[str, Any]) -> str:
    return str(cfg.get("pilot", {}).get("id", "default_pilot"))


def deliverables_dir(cfg: dict[str, Any]) -> Path:
    d = cfg.get("deliverables", {}).get("dir")
    if d:
        return resolve_path(d)
    return resolve_path(f"data/pilot/{pilot_id(cfg)}/deliverables")


def render_template(template_path: Path, context: dict[str, str]) -> str:
    text = template_path.read_text()
    for key, val in context.items():
        text = text.replace(f"{{{{{key}}}}}", val)
    return text


def build_template_context(cfg: dict[str, Any]) -> dict[str, str]:
    pilot = cfg.get("pilot", {})
    targets = cfg.get("targets", {})
    meep = cfg.get("meep", {})
    sb = cfg.get("sim_budget", {})
    today = pilot.get("contract_date") or date.today().isoformat()
    target = float(targets.get("split_ratio_1550", 0.5))
    tol = float(sb.get("tolerance", targets.get("split_ratio_tolerance", 0.05)))
    return {
        "PILOT_ID": pilot_id(cfg),
        "PILOT_TITLE": str(pilot.get("title", "Inverse design pilot")),
        "CLIENT_NAME": str(pilot.get("client_name", "[Client name]")),
        "CONTRACT_DATE": str(today),
        "TARGET_SPLIT": f"{target:.2f}",
        "SPLIT_TOLERANCE": f"{tol:.2f}",
        "WAVELENGTH_NM": str(pilot.get("wavelength_nm", "1550")),
        "PLATFORM": str(pilot.get("platform", "EBL research (drcgenerator manifold)")),
        "RECIPE_VERSION": str(meep.get("recipe_version", "phase0_v1")),
        "MEEP_RESOLUTION": str(meep.get("resolution", 25)),
        "MAX_IL_DB": str(targets.get("max_insertion_loss_db", 12.0)),
        "CONTACT_EMAIL": str(pilot.get("contact_email", "[your email]")),
        "COMPANY_NAME": str(pilot.get("company_name", "[Your company / lab name]")),
        "PILOT_DURATION_WEEKS": str(pilot.get("duration_weeks", 6)),
        "MEEP_BUDGETS": ", ".join(str(b) for b in sb.get("budgets", [30, 50, 100])),
        "IN_SPEC_DEFINITION": (
            f"|split_ratio_upper − {target:.2f}| ≤ {tol:.2f} at the reference wavelength "
            f"under recipe `{meep.get('recipe_version', 'phase0_v1')}`."
        ),
    }


def load_metrics(cfg: dict[str, Any]) -> dict[str, Any]:
    src = cfg.get("sources", {})
    path = resolve_path(src.get("metrics_path", "data/phase1/wedge_a/wedge_a_metrics.json"))
    if not path.exists():
        return {"runs": []}
    return json.loads(path.read_text())


def latest_sim_budget_run(metrics: dict[str, Any]) -> dict[str, Any] | None:
    runs = metrics.get("runs") or []
    if not runs:
        return None
    return runs[-1]


def policy_summary_table(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target = float(run.get("target", 0.5))
    tol = float(run.get("tolerance", 0.05))
    for policy, budgets in (run.get("policies") or {}).items():
        for budget_str, summary in budgets.items():
            rows.append(
                {
                    "policy": policy,
                    "budget": int(budget_str),
                    "target": target,
                    "tolerance": tol,
                    **summary,
                }
            )
    return rows
