#!/usr/bin/env bash
set -euo pipefail
# Legacy wrapper kept for backward compatibility.
# Canonical runner: 5.1-run_fit_nogse_contrast_vs_g.sh with MODEL=rest and APPLY_GRAD_CORR=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=rest \
APPLY_GRAD_CORR=true \
CORR_ROI=Syringe \
ROIS="AntCC,MidAntCC,CentralCC,MidPostCC,PostCC,Left-Lateral-Ventricle,Right-Lateral-Ventricle,Syringe" \
FREE_M0=1.0 \
FIX_D0=3.2e-12 \
PEAK_D0_FIX=3.2e-12 \
bash "$SCRIPT_DIR/5.1-run_fit_nogse_contrast_vs_g.sh" "$@"
