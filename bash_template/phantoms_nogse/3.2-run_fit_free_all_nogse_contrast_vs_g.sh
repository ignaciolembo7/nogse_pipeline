#!/usr/bin/env bash
set -euo pipefail
# Canonical helper: helpers/run_fit_nogse_contrast_vs_g.sh with MODEL=free

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIT_HELPER="$SCRIPT_DIR/helpers/run_fit_nogse_contrast_vs_g.sh"

if [[ ! -f "$FIT_HELPER" ]]; then
    echo "ERROR: fit helper script not found: $FIT_HELPER" >&2
    exit 1
fi

MODEL=free APPLY_GRAD_CORR=false YCOL="value_norm" FIX_M0="1.0" M0_MIN="0.0" M0_MAX="2000.0" bash "$FIT_HELPER" "$@"
