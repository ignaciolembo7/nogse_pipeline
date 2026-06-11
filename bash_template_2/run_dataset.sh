#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"

if [[ $# -lt 1 ]]; then
    echo "ERROR: missing dataset (brains or phantoms)" >&2
    exit 2
fi

DATASET_ARG="$1"
shift

bt2_setup_common
bt2_set_dataset_defaults "$DATASET_ARG"
bt2_run_steps "$@"
