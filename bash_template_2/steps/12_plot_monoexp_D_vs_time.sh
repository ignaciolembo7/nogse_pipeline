#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/lib/common.sh"
bt2_maybe_step_help plot_monoexp_d "$@"
bt2_setup_common
bt2_set_dataset_defaults "${DATASET:?DATASET is required}"

PLOT_MONOEXP_D_SCRIPT="${PLOT_MONOEXP_D_SCRIPT:-$REPO_ROOT/scripts/plotting/plot_monoexp_D_vs_time.py}"
SIGNAL_FITS_ROOT="${SIGNAL_FITS_ROOT:-$ANALYSIS_ROOT/fits/ogse_signal_master}"
MONOEXP_D_OUT_DIR="${MONOEXP_D_OUT_DIR:-$ANALYSIS_ROOT/plots-master/monoexp_D_vs_time}"

bt2_require_file "$PLOT_MONOEXP_D_SCRIPT" "monoexp D plot script"
bt2_require_file "$SIGNAL_FITS_ROOT" "signal fits root"
mkdir -p "$MONOEXP_D_OUT_DIR"

"$PY" "$PLOT_MONOEXP_D_SCRIPT" \
    --fits-root "$SIGNAL_FITS_ROOT" \
    --out-dir "$MONOEXP_D_OUT_DIR" \
    ${PLOT_MONOEXP_D_EXTRA_ARGS:-}
