#!/usr/bin/env bash
set -euo pipefail
# Wrapper for the NOGSE free model with an additive gradient offset g0 and gradient correction.
# Canonical helper: helpers/run_fit_nogse_contrast_vs_g.sh with MODEL=nogse_free_grad_offset

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIT_HELPER="$SCRIPT_DIR/helpers/run_fit_nogse_contrast_vs_g.sh"

if [[ ! -f "$FIT_HELPER" ]]; then
    echo "ERROR: fit helper script not found: $FIT_HELPER" >&2
    exit 1
fi

# Fit controls. Leave the FIX_* variable empty to use the corresponding
# FREE_* initial value. To fix a parameter, set FIX_* and leave FREE_* empty.
#
# Examples:
#   FREE_M0="${FREE_M0:-1.0}" \
# FIX_M0="${FIX_M0:-1.0}" \
#   FIX_M0="" FREE_M0="1.0"      # free M0, seed 1.0
#   FIX_D0="${FIX_D0:-2.3e-12}" \
# FREE_D0="${FREE_D0:-2.3e-12}" \
# FREE_G0="${FREE_G0:-0.0}" \

env \
    MODEL=nogse_free_grad_offset \
    APPLY_GRAD_CORR="${APPLY_GRAD_CORR:-false}" \
    CORR_ROI="${CORR_ROI:-water}" \
    GBASE="${GBASE:-g}" \
    YCOL="${YCOL:-value_norm}" \
    FIX_M0="${FIX_M0:-0.9}" \
    M0_MIN="${M0_MIN:-0.0}" \
    M0_MAX="${M0_MAX:-2000.0}" \
    FREE_D0="${FREE_D0:-2.3e-12}" \
    D0_MIN="${D0_MIN:-}" \
    D0_MAX="${D0_MAX:-}" \
    FIX_G0="${FIX_G0:-0.5}" \
    G0_MIN="${G0_MIN:--20.0}" \
    G0_MAX="${G0_MAX:-20.0}" \
    PEAK_D0_FIX="${PEAK_D0_FIX:-2.3e-12}" \
bash "$FIT_HELPER" "$@"
