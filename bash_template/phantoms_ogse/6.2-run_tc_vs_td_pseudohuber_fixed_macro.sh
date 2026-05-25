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
METHOD="${METHOD:-pseudohuber_fixed_macro}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$PROJECT_ROOT/analysis/phantoms/ogse_experiments}"
CONTRAST_SOURCE="${CONTRAST_SOURCE:-fitted_resampled}" # direct or fitted_resampled
SIGNAL_MODEL="${SIGNAL_MODEL:-rest}"
SIGNAL_G_TYPE="${SIGNAL_G_TYPE:-g}"
if [[ "$CONTRAST_SOURCE" == "fitted_resampled" ]]; then
    FIT_ROOT="${FIT_ROOT:-$ANALYSIS_ROOT/fits/ogse_contrast_vs_gresampled_rest_corr}"
else
    FIT_ROOT="${FIT_ROOT:-$ANALYSIS_ROOT/fits/ogse_contrast_vs_g_rest_corr}"
fi
GROUPFITS="$FIT_ROOT/groupfits_rest.parquet"
SUMMARY_ALPHA="$ANALYSIS_ROOT/alpha_macro/N1/summary_alpha_values.xlsx"
YCOL="${YCOL:-tc_peak_ms}" #tc_fit_ms
EXCLUDE_TD_MS="${EXCLUDE_TD_MS:-209.1}"
SHOW_ERRORBARS="1"
ROIS="${ROIS:-ALL}"
# ROIS="${ROIS:-fiber1,fiber2}"
TD_MIN_MS="0"
TD_MAX_MS="250"
C_FIXED="FREE"
C_MIN="0"
C_MAX="INF"
DELTA_FIXED="FREE"
DELTA_MIN="1e-6"
DELTA_MAX="10000"
EXCLUDE_MATCHES=()
# ------------------------------------------------------------------
# ------------------------------------------------------------------

case "$YCOL" in
    tc_peak_ms)
        TC_DIRNAME="tcpeak_vs_td"
        ;;
    tc_peak_resampled_ms)
        TC_DIRNAME="tcpeak_resampled_vs_td"
        ;;
    tc_fit_ms|tc_ms)
        TC_DIRNAME="tcfit_vs_td"
        ;;
    *)
        TC_DIRNAME="${YCOL}_vs_td"
        ;;
esac
OUT_DIR="$FIT_ROOT/$TC_DIRNAME/$METHOD"

if [[ ! -f "$TC_SCRIPT" ]]; then
    echo "ERROR: Script not found: $TC_SCRIPT" >&2
    exit 1
fi

if [[ ! -f "$GROUPFITS" ]]; then
    echo "Groupfits file not found: $GROUPFITS. Skipping tc vs td."
    exit 0
fi

if [[ ! -f "$SUMMARY_ALPHA" ]]; then
    echo "Summary alpha file not found: $SUMMARY_ALPHA. Skipping tc vs td."
    exit 0
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
if [[ "$ROIS" != "ALL" ]]; then
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
echo "Dataset       : phantoms-3"
echo "Contrast source: $CONTRAST_SOURCE"
echo "ROIs          : $ROIS"
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
