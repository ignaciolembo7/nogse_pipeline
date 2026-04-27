#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PIPELINE="${1:-nogse}"
if (( $# > 0 )); then
    shift
fi

case "${PIPELINE,,}" in
    ogse)
        bash "$SCRIPT_DIR/run_phantoms_pipeline_ogse.sh" "$@"
        ;;
    nogse)
        bash "$SCRIPT_DIR/run_phantoms_pipeline_nogse.sh" "$@"
        ;;
    *)
        echo "ERROR: unknown phantom pipeline '$PIPELINE'." >&2
        echo "Usage: $(basename "$0") [ogse|nogse]" >&2
        exit 1
        ;;
esac
