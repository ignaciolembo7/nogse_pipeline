#!/usr/bin/env bash
set -euo pipefail
export FIT_MODEL="${FIT_MODEL:-ogse_free}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../run_dataset.sh" phantoms fit_contrast "$@"
