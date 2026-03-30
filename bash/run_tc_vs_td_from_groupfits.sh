#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/nogse_pipeline/src:${PYTHONPATH:-}"

PY="${PY:-python}"

# Configuracion
GROUPFITS="$REPO_ROOT/analysis/ogse_experiments/fits/fit_rest_ogse_contrast_rotated_corr/groupfits_rest.parquet"
# METHOD="pseudohuber_free"
METHOD="pseudohuber_fixed_macro"
YCOL="tc_peak_ms"
# YCOL="tc_ms"
FIT_SCRIPT="$REPO_ROOT/nogse_pipeline/scripts/run_tc_vs_td.py"
ALPHA_SUMMARY="$REPO_ROOT/analysis/ogse_experiments/alpha_macro/N1/summary_alpha_values.xlsx"
OUT_DIR="$(dirname "$GROUPFITS")/tc_vs_td/${METHOD}/${YCOL}"
ROIS="ALL"
# ROIS="Left-Lateral-Ventricle,Right-Lateral-Ventricle"

if [[ ! -f "$GROUPFITS" ]]; then
    echo "ERROR: Groupfits file not found: $GROUPFITS" >&2
    exit 1
fi

if [[ ! -f "$FIT_SCRIPT" ]]; then
    echo "ERROR: tc-vs-td script not found: $FIT_SCRIPT" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

extra_args=()
if [[ "$METHOD" == "pseudohuber_fixed_macro" ]]; then
    if [[ ! -f "$ALPHA_SUMMARY" ]]; then
        echo "ERROR: Alpha summary not found: $ALPHA_SUMMARY" >&2
        exit 1
    fi
    extra_args+=(--summary-alpha "$ALPHA_SUMMARY")
fi

if [[ "$ROIS" != "ALL" ]]; then
    read -r -a roi_list <<< "${ROIS//,/ }"
    if (( ${#roi_list[@]} > 0 )); then
        extra_args+=(--rois "${roi_list[@]}")
    fi
fi

echo "============================================================"
echo "Running tc_vs_td"
echo "  Groupfits : $GROUPFITS"
echo "  Method    : $METHOD"
echo "  y-col     : $YCOL"
echo "  ROIs      : $ROIS"
echo "  Out dir   : $OUT_DIR"

"$PY" "$FIT_SCRIPT" \
    --method "$METHOD" \
    --groupfits "$GROUPFITS" \
    --y-col "$YCOL" \
    --out-dir "$OUT_DIR" \
    "${extra_args[@]}"

echo
echo "Finished."
echo "  tc_vs_td out: $OUT_DIR"
