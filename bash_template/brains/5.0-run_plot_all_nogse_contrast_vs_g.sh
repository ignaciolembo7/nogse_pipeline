#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

PLOT_HELPER="$REPO_ROOT/bash_template/helpers/run_plot_contrast_vs_g.sh"

ANALYSIS_ROOT="${ANALYSIS_ROOT:-$PROJECT_ROOT/analysis/brains/ogse_experiments}"
TABLES_ROOT="${TABLES_ROOT:-$ANALYSIS_ROOT/nogse-contrast-data-rotated/tables}"
OUT_ROOT="${OUT_ROOT:-$ANALYSIS_ROOT/nogse-contrast-data-rotated/plots}"
PLOT_SCRIPT="${PLOT_SCRIPT:-$REPO_ROOT/scripts/plot_nogse_contrast_vs_g.py}"
FILE_PATTERN="${FILE_PATTERN:-*.long.parquet}"
XCOL="${XCOL:-g_thorsten_1}"
YCOL="${YCOL:-value_norm}"
STAT="${STAT:-avg}"
DIRECTIONS="${DIRECTIONS:-long,tra}"
ROIS="${ROIS:-Left-Lateral-Ventricle,Right-Lateral-Ventricle,Syringe}"
MISSING_TABLES_MODE="${MISSING_TABLES_MODE:-error}"

if [[ ! -f "$PLOT_HELPER" ]]; then
    echo "ERROR: plot helper script not found: $PLOT_HELPER" >&2
    exit 1
fi

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
PY="${PY:-python}" \
bash "$PLOT_HELPER"
