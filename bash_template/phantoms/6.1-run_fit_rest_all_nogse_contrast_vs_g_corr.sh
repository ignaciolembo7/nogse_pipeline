#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=rest \
APPLY_GRAD_CORR=true \
CORR_ROI=water \
FREE_M0=1.0 \
FIX_D0=2.3e-12 \
PEAK_D0_FIX=2.3e-12 \
bash "$SCRIPT_DIR/5.1-run_fit_nogse_contrast_vs_g.sh" "$@"
