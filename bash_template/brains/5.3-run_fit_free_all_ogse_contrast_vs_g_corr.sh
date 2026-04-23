#!/usr/bin/env bash
set -euo pipefail
# Legacy wrapper kept for backward compatibility.
# Canonical runner: 5.1-run_fit_nogse_contrast_vs_g.sh with MODEL=free and APPLY_GRAD_CORR=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=free APPLY_GRAD_CORR=true CORR_ROI=Syringe bash "$SCRIPT_DIR/5.1-run_fit_nogse_contrast_vs_g.sh" "$@"
