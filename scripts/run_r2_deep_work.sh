#!/usr/bin/env bash
# R² experiment sweep for preprint day — logs to data/phase1/wedge_a/r2_deep_work_log.md
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
source .venv/bin/activate
LOG="data/phase1/wedge_a/r2_deep_work_log.md"
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

if [[ ! -f "$LOG" ]]; then
  echo "# Surrogate R² deep work log" > "$LOG"
  echo "" >> "$LOG"
fi

run_one() {
  local cfg="$1"
  local name="$2"
  echo "==> $name ($cfg)"
  if python scripts/train_wedge_a_surrogate.py --config "$cfg" 2>&1 | tee "/tmp/r2_${name}.log"; then
    local out
    out=$(python -c "import yaml; c=yaml.safe_load(open('$cfg')); print(c['surrogate']['output_dir'])")
    local metrics="$out/metrics.json"
    echo "" >> "$LOG"
    echo "## $(ts) — $name" >> "$LOG"
    echo "- config: \`$cfg\`" >> "$LOG"
    if [[ -f "$metrics" ]]; then
      python -c "
import json
m=json.load(open('$metrics'))
print(f\"- val_r2: {m.get('val_r2', 'n/a'):.4f}\")
print(f\"- val_spearman: {m.get('val_spearman_abs_err', 'n/a')}\")
print(f\"- n_train: {m.get('n_train')} n_val: {m.get('n_val')}\")
" >> "$LOG"
    fi
    local eval_json="data/phase1/wedge_a/ranking_eval_improved.json"
    case "$out" in
      *nearband*) eval_json="data/phase1/wedge_a/ranking_eval.json" ;;
      *abserr*) eval_json="data/phase1/wedge_a/ranking_eval_mask_abserr.json" ;;
    esac
    if [[ -f "$eval_json" ]]; then
      python -c "
import json
e=json.load(open('$eval_json'))
print(f\"- ranking_wins: {e.get('ranking_wins')}\")
print(f\"- spearman_err: {e.get('spearman_err', e.get('val_spearman_abs_err', 'n/a'))}\")
" >> "$LOG" 2>/dev/null || true
    fi
  else
    echo "## $(ts) — $name FAILED" >> "$LOG"
  fi
}

run_one configs/wedge_a_r2_nearband.yaml nearband_0.12
run_one configs/wedge_a_r2_abserr_improved.yaml abserr_rank_boost
run_one configs/wedge_a_r2_joint.yaml joint_rank_mse_0.45
run_one configs/wedge_a_improved.yaml baseline_improved

echo "Done. See $LOG"
