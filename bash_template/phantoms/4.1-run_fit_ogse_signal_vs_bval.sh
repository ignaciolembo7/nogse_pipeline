#!/usr/bin/env bash
set -euo pipefail
# Legacy wrapper kept for backward compatibility.
# Canonical runner: 4.1-run_fit_ogse_signal_vs_g.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/4.1-run_fit_ogse_signal_vs_g.sh" "$@"
