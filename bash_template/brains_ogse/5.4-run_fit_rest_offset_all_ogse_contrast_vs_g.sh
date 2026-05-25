#!/usr/bin/env bash
set -euo pipefail
# Build OGSE contrasts from rest-plus-offset signal fits on a shared resampled gradient grid.
# This does not fit an OGSE contrast model.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAKE_CONTRAST_WRAPPER="$SCRIPT_DIR/3.1-run_make_contrast_selected_rotated.sh"

if [[ ! -f "$MAKE_CONTRAST_WRAPPER" ]]; then
    echo "ERROR: contrast builder not found: $MAKE_CONTRAST_WRAPPER" >&2
    exit 1
fi

export CONTRAST_SOURCE=fitted_resampled
export SIGNAL_MODEL=rest_offset
export SIGNAL_G_TYPE="${SIGNAL_G_TYPE:-g_thorsten}"
export SIGNAL_YCOL="${SIGNAL_YCOL:-value_norm}"
export RESAMPLE_GRID_MIN_MTM="${RESAMPLE_GRID_MIN_MTM:-0}"
export RESAMPLE_GRID_MAX_MTM="${RESAMPLE_GRID_MAX_MTM:-90}"
export RESAMPLE_GRID_N="${RESAMPLE_GRID_N:-1000}"
export SIGNAL_FIX_M0="${SIGNAL_FIX_M0:-1.0}"
export SIGNAL_FREE_M0="${SIGNAL_FREE_M0:-false}"
export SIGNAL_D0_INIT="${SIGNAL_D0_INIT:-0.0032}"
export PEAK_D0_FIX="${PEAK_D0_FIX:-3.2e-12}"

bash "$MAKE_CONTRAST_WRAPPER" "$@"
