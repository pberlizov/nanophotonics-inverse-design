#!/usr/bin/env python3
"""Compare our invrs-gym runs to published leaderboard entries."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "phase1" / "invrs_benchmark"

# Full-fidelity challenge leaderboard (lightweight uses same problem class)
LEADERBOARD_URLS = {
    "ceviche_power_splitter": (
        "https://raw.githubusercontent.com/invrs-io/leaderboard/main/"
        "challenges/ceviche_power_splitter/leaderboard.txt"
    ),
    "ceviche_lightweight_power_splitter": (
        "https://raw.githubusercontent.com/invrs-io/leaderboard/main/"
        "challenges/ceviche_power_splitter/leaderboard.txt"
    ),
}


def fetch_leaderboard(challenge: str) -> list[dict]:
    url = LEADERBOARD_URLS.get(challenge)
    if not url:
        return []
    text = urllib.request.urlopen(url, timeout=30).read().decode()
    rows = []
    for line in text.strip().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = dict(re.findall(r"(\w+)=([^,]+)", line))
        if "eval_metric" in parts:
            rows.append(
                {
                    "path": parts.get("path", "").strip(),
                    "eval_metric": float(parts["eval_metric"]),
                    "minimum_width": float(parts.get("minimum_width", "nan")),
                    "minimum_spacing": float(parts.get("minimum_spacing", "nan")),
                }
            )
    return rows


def best_eval_metric(run: dict) -> tuple[float, str, bool]:
    """Best (lowest) eval_metric; prefer in-spec (eval_metric >= 0)."""
    points: list[tuple[float, str]] = []
    for h in run.get("history") or []:
        if "eval_metric" in h:
            points.append((float(h["eval_metric"]), f"step {h.get('step', '?')}"))
    for key, label in (
        ("best_eval_metric", "best checkpoint"),
        ("final_eval_metric", "final"),
        ("eval_metric", "final"),
    ):
        if key in run and run[key] is not None:
            points.append((float(run[key]), label))
    if not points:
        return float("nan"), "—", False
    in_spec_pts = [(em, lab) for em, lab in points if em >= 0.0]
    pool = in_spec_pts if in_spec_pts else points
    em, lab = min(pool, key=lambda x: x[0])
    return em, lab, em >= 0.0


def load_density_runs(challenge: str) -> list[dict]:
    rows = []
    seen: set[str] = set()
    patterns = sorted(OUT.glob(f"{challenge}*.json"))
    for p in patterns:
        if p.name.startswith("refine") or p.name == "leaderboard_comparison.json":
            continue
        if "baseline" not in p.name and "opt" not in p.name:
            continue
        if p.name in seen:
            continue
        seen.add(p.name)
        d = json.loads(p.read_text())
        if not isinstance(d, dict) or "history" not in d:
            continue
        em, label, in_spec = best_eval_metric(d)
        rows.append(
            {
                "source": p.name,
                "eval_metric": em,
                "metric_note": label,
                "final_eval_metric": d.get("final_eval_metric"),
                "in_spec": in_spec,
                "note": d.get("stopped_reason", ""),
            }
        )
    return rows


def load_refine_runs() -> list[dict]:
    p = OUT / "refine_champion_grad.json"
    if not p.exists():
        return []
    rows = []
    for entry in json.loads(p.read_text()):
        em, label, in_spec = best_eval_metric(entry)
        rows.append(
            {
                "source": f"refine_grad:{entry['sample_id']}",
                "eval_metric": em,
                "metric_note": label,
                "final_eval_metric": entry.get("final_eval_metric"),
                "in_spec": in_spec,
                "note": f"{entry.get('steps', '?')} steps, lr={entry.get('learning_rate', '?')}",
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--challenge", default="ceviche_lightweight_power_splitter")
    p.add_argument(
        "--append-log",
        type=Path,
        default=None,
        help="Append summary row to deep work markdown log",
    )
    args = p.parse_args()

    lb = fetch_leaderboard(args.challenge)
    density = load_density_runs(args.challenge)
    refine = load_refine_runs()
    ours = density + refine

    best_lb = min((r["eval_metric"] for r in lb), default=float("nan"))
    in_spec_ours = [r for r in ours if r.get("in_spec")]
    best_ours = (
        min(r["eval_metric"] for r in in_spec_ours)
        if in_spec_ours
        else float("nan")
    )

    lines = [
        "# invrs-gym leaderboard comparison",
        "",
        f"Challenge: `{args.challenge}`",
        f"Published entries: {len(lb)}",
        "",
        "_Lower `eval_metric` is better. In-spec: eval_metric ≥ 0._",
        "",
        "## Published (best first)",
        "",
        "| eval_metric | min_width | min_spacing | path |",
        "|-------------|-----------|-------------|------|",
    ]
    for r in sorted(lb, key=lambda x: x["eval_metric"])[:10]:
        path = r["path"].split("/")[-1] if r["path"] else "—"
        lines.append(
            f"| {r['eval_metric']:.4f} | {r['minimum_width']:.1f} | "
            f"{r['minimum_spacing']:.1f} | {path} |"
        )

    lines += [
        "",
        "## Ours — gym-native density opt",
        "",
        "| source | best eval_metric | (final) | in_spec | note |",
        "|--------|------------------|---------|---------|------|",
    ]
    for r in density:
        flag = "✓" if r.get("in_spec") else "—"
        fin = r.get("final_eval_metric")
        fin_s = f"{fin:.4f}" if fin is not None else "—"
        lines.append(
            f"| {r['source']} | {r['eval_metric']:.4f} ({r['metric_note']}) | "
            f"{fin_s} | {flag} | {r.get('note', '')} |"
        )

    lines += [
        "",
        "## Ours — champion latent refine (Path A)",
        "",
        "| source | best eval_metric | (final) | in_spec | note |",
        "|--------|------------------|---------|---------|------|",
    ]
    if refine:
        for r in refine:
            flag = "✓" if r.get("in_spec") else "—"
            fin = r.get("final_eval_metric")
            fin_s = f"{fin:.4f}" if fin is not None else "—"
            lines.append(
                f"| {r['source']} | {r['eval_metric']:.4f} ({r['metric_note']}) | "
                f"{fin_s} | {flag} | {r.get('note', '')} |"
            )
    else:
        lines.append("| _none_ | — | — | — | run `refine_champion_grad.py` |")

    if in_spec_ours and lb:
        gap = best_ours - best_lb
        lines += [
            "",
            f"**Best published eval_metric:** {best_lb:.4f}",
            f"**Best ours (in-spec only):** {best_ours:.4f}",
            f"**Gap to published top:** {gap:+.4f} "
            f"({'behind' if gap > 0 else 'ahead' if gap < 0 else 'tie'})",
        ]

    OUT.mkdir(parents=True, exist_ok=True)
    report = OUT / "leaderboard_comparison.md"
    report.write_text("\n".join(lines) + "\n")
    summary = {
        "challenge": args.challenge,
        "n_published": len(lb),
        "best_published_eval_metric": best_lb,
        "best_ours_eval_metric": best_ours,
        "gap_to_published": best_ours - best_lb if ours and lb else None,
        "density_runs": density,
        "refine_runs": refine,
    }
    (OUT / "leaderboard_comparison.json").write_text(json.dumps(summary, indent=2))
    if args.append_log:
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        log = args.append_log
        log.parent.mkdir(parents=True, exist_ok=True)
        if not log.exists():
            log.write_text("# invrs-gym deep work log\n\n")
        with log.open("a") as f:
            f.write(
                f"\n## {ts} — {args.challenge}\n"
                f"- best published: {best_lb:.4f}\n"
                f"- best ours (in-spec): {best_ours:.4f}\n"
                f"- gap: {best_ours - best_lb:+.4f}\n"
                f"- density runs: {len(density)} · refine runs: {len(refine)}\n"
            )
    print(report.read_text())
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
