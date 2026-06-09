#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"
RUNNER_LIB="$REPO_ROOT/bash_template/helpers/pipeline_runner_lib.sh"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DEFAULT_PY="python"
PROJECT_PY="/home/ignacio.lemboferrari@unitn.it/.conda/envs/nogse_pipe_env/bin/python"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    DEFAULT_PY="${CONDA_PREFIX}/bin/python"
elif [[ -x "$PROJECT_PY" ]]; then
    DEFAULT_PY="$PROJECT_PY"
elif command -v python3 >/dev/null 2>&1; then
    DEFAULT_PY="$(command -v python3)"
fi
PY="${PY:-$DEFAULT_PY}"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/phantoms_ogse}"
ONEG="${ONEG:-true}"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

if [[ ! -f "$RUNNER_LIB" ]]; then
    echo "ERROR: pipeline runner helper not found: $RUNNER_LIB" >&2
    exit 1
fi

source "$RUNNER_LIB"

RUN_SCRIPTS=(
# 20220610-PHANTOM3 OGSE workflow, aligned with brains_ogse from 2.0 onward.
# "0.0-run_dicom2nifti.sh"
# DICOM metadata extraction is centralized in bash_template/dicom_params.
# "0.1-run_make_gval_gvec.sh"
# "0.2-prep_phantom_b0.sh"
# "0.3-copy_selected_files.sh"
# "1.0-run_PHANTOM-denoised_signal_extraction.sh"
# "2.0-run_process_all_results.sh"
# "3.1-run_make_contrast_selected.sh"
# "3.2-run_plot_all_ogse_contrast_vs_g.sh"
# "3.3-run_make_alpha_macro_summary.sh"
# "4.1-run_fit_ogse_signal_vs_g.sh"
# "4.2-run_plot_monoexp_D_vs_time.sh"
# "4.3-run_plot_D0_vs_Delta.sh"
# "5.1-run_fit_free_ogse_contrast_vs_g.sh"
# "5.2-run_make_grad_correction_table.sh"
# "5.3-run_fit_free_all_ogse_contrast_vs_g.sh"
# "5.4-run_fit_rest_all_ogse_contrast_vs_g.sh"
# "5.4-run_fit_rest_offset_globC_ogse_contrast_vs_g.sh"
# "5.5-run_fit_mixed_global_ogse_contrast_vs_g.sh"
"6.1-run_make_groupfits_rest.sh"
"6.2-run_tc_vs_td_pseudohuber_fixed_macro.sh"
)

run_pipeline_steps \
    "Phantoms OGSE pipeline runner" \
    "$SCRIPT_DIR" \
    "$PROJECT_ROOT" \
    "$REPO_ROOT" \
    "$LOG_ROOT" \
    "$PY" \
    "$ONEG" \
    RUN_SCRIPTS
