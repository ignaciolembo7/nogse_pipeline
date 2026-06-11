#!/usr/bin/env bash
set -euo pipefail
export SIGNAL_FIT_MODEL="${SIGNAL_FIT_MODEL:-monoexp}"
export SIGNAL_FIT_G_TYPE="${SIGNAL_FIT_G_TYPE:-bvalue_thorsten}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../run_dataset.sh" phantoms fit_signal "$@"
