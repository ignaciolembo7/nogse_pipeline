#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
PY="${PY:-python}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/nogse_pipeline/logs/brains}"

RUN_SCRIPTS=(
#   "0.0-run_dicom2nifti.sh"    
#   "1-run_BRAINS-denoised_topup_signal_extraction.sh"
#   "2.0-run_process_all_results.sh" # starting from this step, results will be saved in analysis/brains
#   "3.0-run_rotate_all_signals.sh"
#   "3.1-run_make_contrast_selected_rotated.sh"
#   "3.2-run_plot_all_ogse_contrast_vs_g.sh"
#   "4.1-run_fit_ogse_signal_vs_bval.sh"
#   "4.2-run_plot_monoexp_D_vs_time.sh"
#   "4.3-run_make_alpha_macro_summary.sh"
#   "4.4-run_plot_D0_vs_Delta.sh"
#   "5.1-run_fit_free_all_ogse_contrast_vs_g.sh"
#   "5.2-run_make_grad_correction_table.sh"
#   "5.3-run_fit_free_all_ogse_contrast_vs_g_corr.sh"
#   "6.1-run_fit_rest_all_ogse_contrast_vs_g_corr.sh"
  "6.2-run_make_groupfits_rest.sh"
  "6.3-run_tc_vs_td_pseudohuber_fixed_macro.sh"
)

mkdir -p "$LOG_ROOT"

echo "============================================================"
echo "Brains pipeline runner"
echo "Script dir : $SCRIPT_DIR"
echo "Project    : $PROJECT_ROOT"
echo "PY         : $PY"
echo "Log root   : $LOG_ROOT"
echo "============================================================"

total=0
ok=0
failed=0
declare -a failed_steps=()

for script_name in "${RUN_SCRIPTS[@]}"; do
    total=$((total + 1))
    script_path="$SCRIPT_DIR/$script_name"
    log_path="$LOG_ROOT/${script_name%.sh}.log"

    echo
    echo "============================================================"
    echo "Step $total"
    echo "Script : $script_name"
    echo "Log    : $log_path"
    echo "============================================================"

    if [[ ! -f "$script_path" ]]; then
        failed=$((failed + 1))
        failed_steps+=("missing :: $script_name")
        echo "ERROR: script not found: $script_path" >&2
        continue
    fi

    if PY="$PY" bash "$script_path" >"$log_path" 2>&1; then
        ok=$((ok + 1))
        echo "OK: $script_name"
    else
        status=$?
        failed=$((failed + 1))
        failed_steps+=("exit $status :: $script_name")
        echo "WARNING: $script_name failed with exit code $status" >&2
        echo "Check log: $log_path" >&2
    fi
done

echo
echo "============================================================"
echo "Finished."
echo "  Total steps : $total"
echo "  Successful  : $ok"
echo "  Failed      : $failed"

if (( failed > 0 )); then
    echo
    echo "Failed steps:"
    for item in "${failed_steps[@]}"; do
        echo "  - $item"
    done
fi
