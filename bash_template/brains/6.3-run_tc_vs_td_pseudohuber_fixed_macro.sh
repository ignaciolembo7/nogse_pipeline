#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

PY="${PY:-python}"
TC_SCRIPT="$REPO_ROOT/scripts/run_tc_vs_td.py"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

METHOD="pseudohuber_fixed_macro"
GROUPFITS="$PROJECT_ROOT/analysis/brains/ogse_experiments/fits/fit_rest_ogse_contrast_rotated_corr/groupfits_rest.parquet"
SUMMARY_ALPHA="$PROJECT_ROOT/analysis/brains/ogse_experiments/alpha_macro/N1/summary_alpha_values.xlsx"
YCOL="tc_peak_ms"
EXCLUDE_TD_MS="76"
SHOW_ERRORBARS="1"
ROIS="AntCC,MidAntCC,CentralCC,MidPostCC,PostCC,Left-Lateral-Ventricle,Right-Lateral-Ventricle,Syringe"
TD_MIN_MS="0"
TD_MAX_MS="250"
C_FIXED="FREE"
C_MIN="0"
C_MAX="10"
DELTA_FIXED="FREE"
DELTA_MIN="1e-6"
DELTA_MAX="100"
EXCLUDE_MATCHES=()
if [[ "$YCOL" == "tc_peak_ms" ]]; then
    TC_DIRNAME="tcpeak_vs_td"
elif [[ "$YCOL" == "tc_fit_ms" || "$YCOL" == "tc_ms" ]]; then
    TC_DIRNAME="tcfit_vs_td"
else
    TC_DIRNAME="${YCOL}_vs_td"
fi
OUT_DIR="$PROJECT_ROOT/analysis/brains/ogse_experiments/fits/fit_rest_ogse_contrast_rotated_corr/$TC_DIRNAME/$METHOD" #/$YCOL"

if [[ ! -f "$TC_SCRIPT" ]]; then
    echo "ERROR: Script not found: $TC_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$GROUPFITS" ]]; then
    echo "ERROR: Groupfits file not found: $GROUPFITS" >&2
    exit 1
fi

if [[ ! -f "$SUMMARY_ALPHA" ]]; then
    echo "ERROR: Summary alpha file not found: $SUMMARY_ALPHA" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

extra_args=()
if [[ -n "${EXCLUDE_TD_MS// }" ]]; then
    read -r -a exclude_td_list <<< "${EXCLUDE_TD_MS//,/ }"
    if (( ${#exclude_td_list[@]} > 0 )); then
        extra_args+=(--exclude-td-ms "${exclude_td_list[@]}")
    fi
fi
if (( ${#EXCLUDE_MATCHES[@]} > 0 )); then
    extra_args+=(--exclude-match "${EXCLUDE_MATCHES[@]}")
fi
if [[ "${SHOW_ERRORBARS}" == "0" || "${SHOW_ERRORBARS,,}" == "false" || "${SHOW_ERRORBARS,,}" == "no" ]]; then
    extra_args+=(--no-errorbars)
fi
if [[ -n "${ROIS// }" ]]; then
    read -r -a roi_list <<< "${ROIS//,/ }"
    if (( ${#roi_list[@]} > 0 )); then
        extra_args+=(--rois "${roi_list[@]}")
    fi
fi
extra_args+=(--td-min-ms "$TD_MIN_MS" --td-max-ms "$TD_MAX_MS")
extra_args+=(--c-min "$C_MIN" --c-max "$C_MAX")
extra_args+=(--delta-min "$DELTA_MIN" --delta-max "$DELTA_MAX")
if [[ "$C_FIXED" != "FREE" ]]; then
    extra_args+=(--c-fixed "$C_FIXED")
fi
if [[ "$DELTA_FIXED" != "FREE" ]]; then
    extra_args+=(--delta-fixed "$DELTA_FIXED")
fi

echo "============================================================"
echo "Dataset       : brains"
echo "ROIs           : $ROIS"
echo "Method        : $METHOD"
echo "Groupfits     : $GROUPFITS"
echo "Summary alpha : $SUMMARY_ALPHA"
echo "Y column      : $YCOL"
echo "Exclude td_ms : ${EXCLUDE_TD_MS:-<none>}"
echo "Exclude rows  : ${EXCLUDE_MATCHES[*]:-<none>}"
echo "Error bars    : $SHOW_ERRORBARS"
echo "Td x limits   : $TD_MIN_MS $TD_MAX_MS"
echo "c fit control : fixed=$C_FIXED bounds=[$C_MIN, $C_MAX]"
echo "delta control : fixed=$DELTA_FIXED bounds=[$DELTA_MIN, $DELTA_MAX]"
echo "tc_vs_td kind : $TC_DIRNAME"
echo "Output dir    : $OUT_DIR"

"$PY" "$TC_SCRIPT" \
    --method "$METHOD" \
    --groupfits "$GROUPFITS" \
    --summary-alpha "$SUMMARY_ALPHA" \
    --y-col "$YCOL" \
    --out-dir "$OUT_DIR" \
    "${extra_args[@]}"

echo
echo "Finished."
