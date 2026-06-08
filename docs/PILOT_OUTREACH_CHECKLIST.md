# Outreach checklist (print or pin)

## Before first email

- [ ] Edit `configs/pilot/benchmark_50_50.yaml`: `contact_email`, `company_name`
- [ ] Run `bash scripts/run_pilot.sh --config configs/pilot/benchmark_50_50.yaml --report-only`
- [ ] Open `data/pilot/benchmark_50_50/outreach/ONE_PAGER.md` — sanity read
- [ ] Optional: `bash scripts/run_pilot.sh ... --full-meep --skip-train` for B=30/50/100 curves
- [ ] Pick 2 PNGs from `deliverables/designs/*/`

## Email attachments

1. `ONE_PAGER.md` (or PDF export)
2. `sim_budget_curve.png`
3. One champion mask PNG (`local_00022` recommended)

## First call — ask

1. Target split, λ, bandwidth, IL budget
2. EBL vs photolithography; rule deck?
3. How many FDTD runs today per acceptable design?
4. MPW timeline?

## Do not claim

- Fab-guaranteed performance
- PDK match without calibration SOW
- Surrogate-only sign-off

## After verbal yes

1. Copy `configs/pilot/client_template.yaml` → `configs/pilot/<client>.yaml`
2. Send `PILOT_OFFER.md` + `PILOT_SOW.md` + filled `SIM_CONTRACT.md`
3. Kickoff → `bash scripts/run_pilot.sh --config configs/pilot/<client>.yaml`
