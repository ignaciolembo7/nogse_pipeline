#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

PLOT_HELPER="$REPO_ROOT/bash_template/helpers/run_plot_contrast_vs_g.sh"

PY="${PY:-python}"
# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$PROJECT_ROOT/analysis/phantoms/nogse_experiments}"
TABLES_ROOT="${TABLES_ROOT:-$ANALYSIS_ROOT/contrast_data/tables}"
OUT_ROOT="${OUT_ROOT:-$ANALYSIS_ROOT/contrast_data/plots}"
PLOT_SCRIPT="${PLOT_SCRIPT:-$REPO_ROOT/scripts/plot_nogse_contrast_vs_g.py}"
FILE_PATTERN="${FILE_PATTERN:-*.long.parquet}"
XCOL="${XCOL:-g_1}"
YCOL="${YCOL:-value_norm}"
STAT="${STAT:-avg}"
DIRECTIONS="${DIRECTIONS:-ALL}"
ROIS="${ROIS:-ALL}"
MISSING_TABLES_MODE="${MISSING_TABLES_MODE:-skip}"
# ------------------------------------------------------------------
# ------------------------------------------------------------------

if [[ ! -f "$PLOT_HELPER" ]]; then
    echo "ERROR: plot helper script not found: $PLOT_HELPER" >&2
    exit 1
fi

PY="$PY" \
ANALYSIS_ROOT="$ANALYSIS_ROOT" \
TABLES_ROOT="$TABLES_ROOT" \
OUT_ROOT="$OUT_ROOT" \
PLOT_SCRIPT="$PLOT_SCRIPT" \
FILE_PATTERN="$FILE_PATTERN" \
XCOL="$XCOL" \
YCOL="$YCOL" \
STAT="$STAT" \
DIRECTIONS="$DIRECTIONS" \
ROIS="$ROIS" \
MISSING_TABLES_MODE="$MISSING_TABLES_MODE" \
bash "$PLOT_HELPER"
