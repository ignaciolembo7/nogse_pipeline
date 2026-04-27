#!/usr/bin/env bash
set -euo pipefail
# Canonical helper: helpers/run_fit_nogse_contrast_vs_g.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIT_HELPER="$SCRIPT_DIR/helpers/run_fit_nogse_contrast_vs_g.sh"

if [[ ! -f "$FIT_HELPER" ]]; then
    echo "ERROR: fit helper script not found: $FIT_HELPER" >&2
    exit 1
fi

bash "$FIT_HELPER" "$@"
