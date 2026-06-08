# Pilot configs

| File | Purpose |
|------|---------|
| `benchmark_50_50.yaml` | Internal benchmark for outreach collateral (50/50 splitter) |
| `client_template.yaml` | Copy and rename for a paying customer |

Each file **extends** `configs/wedge_a.yaml` via `base_config`. Override only what differs:

- `pilot.*` — client-facing metadata
- `targets.*` — split ratio, tolerance, IL cap
- `deliverables.*` — output folder, top-k, GDS
- `champions` — pre-verified designs to include in dossier
- `sources.*` — paths to metrics / sim-budget results

**Run**

```bash
# Full pipeline (train + MEEP sim-budget + deliverables + report)
bash scripts/run_pilot.sh --config configs/pilot/benchmark_50_50.yaml

# Collateral only from existing wedge_a metrics (fast)
bash scripts/run_pilot.sh --config configs/pilot/benchmark_50_50.yaml --report-only

# Full sim-budget budgets 30/50/100 (hours of MEEP)
bash scripts/run_pilot.sh --config configs/pilot/benchmark_50_50.yaml --full-meep
```

See [docs/PILOT_README.md](../../docs/PILOT_README.md).
