#!/usr/bin/env bash
set -euo pipefail
# Legacy-style free wrapper for uncorrected OGSE contrast fits.
# Canonical helper: helpers/run_fit_ogse_contrast_vs_g.sh with MODEL=free and APPLY_GRAD_CORR=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIT_HELPER="$SCRIPT_DIR/helpers/run_fit_ogse_contrast_vs_g.sh"

if [[ ! -f "$FIT_HELPER" ]]; then
    echo "ERROR: fit helper script not found: $FIT_HELPER" >&2
    exit 1
fi
# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODEL=free \
APPLY_GRAD_CORR=false \
CORR_ROI=water \
GBASE=g \
PEAK_D0_FIX=2.3e-12 \
bash "$FIT_HELPER" "$@"
# ------------------------------------------------------------------
# ------------------------------------------------------------------
