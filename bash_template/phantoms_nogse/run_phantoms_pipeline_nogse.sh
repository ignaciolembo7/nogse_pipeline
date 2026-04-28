#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"
RUNNER_LIB="$REPO_ROOT/bash_template/helpers/pipeline_runner_lib.sh"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
PY="${PY:-python}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/nogse_pipeline/logs/phantoms-2_nogse}"
ONEG="${ONEG:-true}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

RUN_SCRIPTS=(
# Common setup / extraction steps
# "0.0-run_dicom2nifti.sh"
# "0.1-run_make_gval_gvec.sh"
# "0.2-prep_phantom_b0.sh"
# "0.3-copy_selected_files.sh"
# "1.0-run_PHANTOM-denoised_signal_extraction.sh"
# "1.1-run_plot_signal_image_grid.sh"
# "2.0-run_process_all_results.sh"
# "2.1-run_plot_selected_nogse_signals.sh"
# "2.2-run_fit_nogse_signal_vs_g.sh"

# NOGSE contrast analysis
# "3.0-run_make_nogse_contrast_selected.sh"
# "3.1-run_plot_all_nogse_contrast_vs_g.sh"
"3.2-run_fit_free_all_nogse_contrast_vs_g.sh"
# "5.2-run_make_grad_correction_table.sh"
# "5.3-run_fit_free_all_nogse_contrast_vs_g_corr.sh"
# "6.1-run_fit_rest_all_nogse_contrast_vs_g_corr.sh"
# "6.2-run_make_groupfits_rest.sh"
# "6.3-run_tc_vs_td_pseudohuber_fixed_macro.sh"
)
# ------------------------------------------------------------------
# ------------------------------------------------------------------

if [[ ! -f "$RUNNER_LIB" ]]; then
    echo "ERROR: pipeline runner helper not found: $RUNNER_LIB" >&2
    exit 1
fi

source "$RUNNER_LIB"

run_pipeline_steps \
    "Phantoms NOGSE pipeline runner" \
    "$SCRIPT_DIR" \
    "$PROJECT_ROOT" \
    "$REPO_ROOT" \
    "$LOG_ROOT" \
    "$PY" \
    "$ONEG" \
    RUN_SCRIPTS
