#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

PY="${PY:-python}"
HELPER_SCRIPT="$REPO_ROOT/bash_template/helpers/run_make_nogse_contrast_auto.sh"

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/analysis/brains/ogse_experiments/data-rotated}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/analysis/brains/ogse_experiments/nogse-contrast-data-rotated}"
DIRECTIONS="${DIRECTIONS:-long,tra}"
ONEG="${ONEG:-false}"
SUBJS="${SUBJS:-ALL}"

if [[ ! -f "$HELPER_SCRIPT" ]]; then
    echo "ERROR: helper script not found: $HELPER_SCRIPT" >&2
    exit 1
fi

PY="$PY" \
DATA_ROOT="$DATA_ROOT" \
OUT_ROOT="$OUT_ROOT" \
DIRECTIONS="$DIRECTIONS" \
ONEG="$ONEG" \
SUBJS="$SUBJS" \
bash "$HELPER_SCRIPT"
