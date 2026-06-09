#!/usr/bin/env bash
set -euo pipefail
# Rest-plus-offset wrapper with a global signal-fit C for corrected OGSE contrast fits.
# Canonical helper: helpers/run_fit_ogse_contrast_vs_g.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIT_HELPER="$SCRIPT_DIR/helpers/run_fit_ogse_contrast_vs_g.sh"

if [[ ! -f "$FIT_HELPER" ]]; then
    echo "ERROR: fit helper script not found: $FIT_HELPER" >&2
    exit 1
fi
# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
export MODEL=rest_offset_globC
export SIGNAL_MODEL="${SIGNAL_MODEL:-rest_offset_globC}"
export CONTRAST_SOURCE="${CONTRAST_SOURCE:-fitted_resampled}"
export APPLY_GRAD_CORR=true
export CORR_ROI=Syringe
export EXPORT_FIT_PARAMS_PATTERN="${EXPORT_FIT_PARAMS_PATTERN:-**/fit_params.rest_offset_globC.g_thorsten.value_norm.direction_*.parquet}"
export RESAMPLED_GRID_MIN_MTM="${RESAMPLED_GRID_MIN_MTM:-0}"
export RESAMPLED_GRID_MAX_MTM="${RESAMPLED_GRID_MAX_MTM:-90}"
export RESAMPLED_GRID_N="${RESAMPLED_GRID_N:-1000}"

# ROIs to fit. Use ALL to keep every ROI in the input tables.
export ROIS="AntCC,MidAntCC,CentralCC,MidPostCC,PostCC"

# M0 mode. Defaults to fixed M0=1. Set FREE_M0=1.0 to fit M0.
export FIX_M0="${FIX_M0:-1.0}"
export FREE_M0="${FREE_M0:-}"

# D0 mode. D0 is in m^2/ms. Keep one block active.
export FIX_D0=3.2e-12
export FREE_D0=

# tc mode. tc is in ms. Keep one block active.
export FIX_TC=
export FREE_TC=5.0

# C mode. C is fitted globally across the two signal curves during export.
export FIX_C=
export FREE_C=0.0

# Fit bounds. Each variable is "MIN MAX".
export M0_BOUNDS="0.0 2.0"
export D0_BOUNDS="2.3e-14 2.3e-10"
export TC_BOUNDS="0.1 1000.0"
export C_BOUNDS="0.0 1.0"

# D0 used only to convert the fitted contrast peak into tc_peak_ms.
export PEAK_D0_FIX="3.2e-12"

# Raw x-axis gradient range used to search tc_peak on the fitted contrast curve.
export PEAK_G_MAX_MTM="1000"

bash "$FIT_HELPER" "$@"
# ------------------------------------------------------------------
# ------------------------------------------------------------------
