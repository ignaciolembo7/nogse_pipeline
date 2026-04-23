#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL=free APPLY_GRAD_CORR=false bash "$SCRIPT_DIR/5.1-run_fit_nogse_contrast_vs_g.sh" "$@"
