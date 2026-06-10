#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIT_HELPER="$SCRIPT_DIR/../helpers/run_fit_global_signal.sh"

export FAMILY="${FAMILY:-nogse}"
export DATASET_KIND="${DATASET_KIND:-brains}"

if [[ ! -f "$FIT_HELPER" ]]; then
    echo "ERROR: global signal fit helper not found: $FIT_HELPER" >&2
    exit 1
fi

bash "$FIT_HELPER"
