#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

PY="${PY:-python}"
TC_SCRIPT="$REPO_ROOT/scripts/run_tc_vs_td.py"

METHOD="pseudohuber_fixed_macro"
GROUPFITS="$PROJECT_ROOT/analysis/brains/ogse_experiments/fits/fit_rest_ogse_contrast_rotated_corr/groupfits_rest.parquet"
SUMMARY_ALPHA="$PROJECT_ROOT/analysis/brains/ogse_experiments/alpha_macro/N1/summary_alpha_values.xlsx"
YCOL="tc_peak_ms"
EXCLUDE_TD_MS="209.1"
SHOW_ERRORBARS="0"
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

echo "============================================================"
echo "Dataset       : brains"
echo "Method        : $METHOD"
echo "Groupfits     : $GROUPFITS"
echo "Summary alpha : $SUMMARY_ALPHA"
echo "Y column      : $YCOL"
echo "Exclude td_ms : ${EXCLUDE_TD_MS:-<none>}"
echo "Exclude rows  : ${EXCLUDE_MATCHES[*]:-<none>}"
echo "Error bars    : $SHOW_ERRORBARS"
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
