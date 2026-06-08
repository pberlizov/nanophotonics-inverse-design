#!/usr/bin/env python3
"""C-band split ratio: narrowband champions vs broadband-aware search best."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / "data/phase1/release"
OUT = REPO / "docs/preprint/figures"

GATE_LO, GATE_HI = 0.45, 0.55


def load_champions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [r for r in data if r.get("status") == "ok" and r.get("R_up_by_wl")]
    return []


def load_hunt_best(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    candidates: list[dict] = []
    for sec in payload.get("sections", []):
        if sec.get("name") != "verify":
            continue
        data = sec.get("data") or {}
        for r in data.get("results", []):
            if r.get("status") == "ok" and r.get("R_up_by_wl"):
                candidates.append(r)
    if not candidates:
        for r in payload.get("winners", []):
            if r.get("R_up_by_wl"):
                candidates.append(r)
    if not candidates:
        return None
    return min(candidates, key=lambda r: float(r.get("worst_split_error", 999)))


def series_from_record(r: dict) -> tuple[np.ndarray, np.ndarray]:
    by_wl = r.get("R_up_by_wl") or {}
    xs = sorted(float(k) for k in by_wl.keys())
    ys = [float(by_wl.get(str(x), by_wl.get(x))) for x in xs]
    return np.asarray(xs), np.asarray(ys)


def shade_gate(ax) -> None:
    ax.axhspan(GATE_LO, GATE_HI, color="#22c55e", alpha=0.12, zorder=0)
    ax.axhline(0.5, color="#64748b", ls="--", lw=0.9, zorder=1)
    ax.axhline(GATE_LO, color="#94a3b8", ls=":", lw=0.8, zorder=1)
    ax.axhline(GATE_HI, color="#94a3b8", ls=":", lw=0.8, zorder=1)


def main() -> None:
    narrow = load_champions(RELEASE / "champion_broadband.json")
    hunt_best = load_hunt_best(RELEASE / "broadband_hunt.json")
    if not narrow:
        raise SystemExit(f"missing champion curves in {RELEASE / 'champion_broadband.json'}")

    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    all_x: list[float] = []
    for r in narrow[:5]:
        xs, ys = series_from_record(r)
        all_x.extend(xs.tolist())
        ax.plot(xs, ys, marker="o", ms=4, lw=1.8, alpha=0.85, label=r.get("sample_id", "?"))

    if hunt_best:
        xs, ys = series_from_record(hunt_best)
        all_x.extend(xs.tolist())
        sid = hunt_best.get("sample_id", "broadband hunt best")
        ax.plot(
            xs,
            ys,
            ls="--",
            lw=2.6,
            color="#dc2626",
            marker="s",
            ms=4,
            label=f"{sid} (broadband hunt best, fails gate)",
        )

    shade_gate(ax)
    xlo = min(all_x) if all_x else 1.53
    xhi = max(all_x) if all_x else 1.57
    pad = max(0.005, 0.02 * (xhi - xlo))
    ax.set_xlim(xlo - pad, xhi + pad)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Wavelength $\lambda$ (µm)")
    ax.set_ylabel(r"$R_{\mathrm{up}}$")
    ax.set_title("C-band split ratio: narrowband champions vs broadband-aware search")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.92)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"broadband_contribution.{ext}", dpi=200, bbox_inches="tight")
    release = REPO / "data/phase1/release"
    for ext in ("png", "pdf"):
        src = OUT / f"broadband_contribution.{ext}"
        (release / f"champion_broadband_contribution.{ext}").write_bytes(src.read_bytes())
    plt.close()
    print(f"wrote {OUT / 'broadband_contribution.png'}")


if __name__ == "__main__":
    main()
