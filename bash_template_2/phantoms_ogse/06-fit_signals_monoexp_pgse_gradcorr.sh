#!/usr/bin/env bash
set -euo pipefail
export SIGNAL_FIT_MODEL="${SIGNAL_FIT_MODEL:-monoexp}"
export SIGNAL_FIT_G_TYPE="${SIGNAL_FIT_G_TYPE:-bvalue_thorsten}"
export CORR_ROI="${CORR_ROI:-water}"
export CORR_XLSX="${CORR_XLSX:-analysis/phantoms/ogse_experiments/fits/grad_correction_master/${CORR_ROI}.grad_correction.xlsx}"
export SIGNAL_FIT_EXTRA_ARGS="${SIGNAL_FIT_EXTRA_ARGS:-} --apply_grad_corr --corr_xlsx $CORR_XLSX --corr_roi $CORR_ROI"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../run_dataset.sh" phantoms fit_signal "$@"
