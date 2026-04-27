#!/usr/bin/env bash
set -euo pipefail
# Legacy wrapper kept for backward compatibility.
# Canonical helper: helpers/run_fit_nogse_contrast_vs_g.sh with MODEL=rest and APPLY_GRAD_CORR=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIT_HELPER="$SCRIPT_DIR/helpers/run_fit_nogse_contrast_vs_g.sh"

if [[ ! -f "$FIT_HELPER" ]]; then
    echo "ERROR: fit helper script not found: $FIT_HELPER" >&2
    exit 1
fi

MODEL=rest \
APPLY_GRAD_CORR=true \
CORR_ROI=water \
FREE_M0=1.0 \
FIX_D0=2.3e-12 \
PEAK_D0_FIX=2.3e-12 \
bash "$FIT_HELPER" "$@"
