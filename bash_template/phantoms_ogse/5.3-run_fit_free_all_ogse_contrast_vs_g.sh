#!/usr/bin/env bash
set -euo pipefail
# Free-model wrapper for corrected OGSE contrast fits.
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
export MODEL=free
export APPLY_GRAD_CORR=true
export CORR_ROI=water
export GBASE=g

# ROIs to fit. Use ALL to keep every ROI in the input tables.
export ROIS="ALL"

# M0 mode. Keep one block active.
export FIX_M0=1.0
export FREE_M0=

# D0 mode. D0 is in m^2/ms. Keep one block active.
export FIX_D0=
export FREE_D0=2.3e-12

# Fit bounds. Each variable is "MIN MAX".
export M0_BOUNDS="0.0 2.0"
export D0_BOUNDS="2.3e-14 2.3e-10"

# D0 used only to convert the fitted contrast peak into tc_peak_ms.
export PEAK_D0_FIX="2.3e-12"

bash "$FIT_HELPER" "$@"
# ------------------------------------------------------------------
# ------------------------------------------------------------------
