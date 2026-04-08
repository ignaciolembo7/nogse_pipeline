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
# YCOL="tc_fit_ms"
EXCLUDE_TD_MS="209.1"
SHOW_ERRORBARS="0"
EXCLUDE_MATCHES=()
FIT_SCRIPT="$REPO_ROOT/nogse_pipeline/scripts/run_tc_vs_td.py"
ALPHA_SUMMARY="$REPO_ROOT/analysis/ogse_experiments/alpha_macro/N1/summary_alpha_values.xlsx"
if [[ "$YCOL" == "tc_peak_ms" ]]; then
    TC_DIRNAME="tcpeak_vs_td"
elif [[ "$YCOL" == "tc_fit_ms" || "$YCOL" == "tc_ms" ]]; then
    TC_DIRNAME="tcfit_vs_td"
else
    TC_DIRNAME="${YCOL}_vs_td"
fi
OUT_DIR="$(dirname "$GROUPFITS")/${TC_DIRNAME}/${METHOD}/${YCOL}"
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
echo "Running tc_vs_td"
echo "  Groupfits : $GROUPFITS"
echo "  Method    : $METHOD"
echo "  y-col     : $YCOL"
echo "  excl td   : ${EXCLUDE_TD_MS:-<none>}"
echo "  excl rows : ${EXCLUDE_MATCHES[*]:-<none>}"
echo "  err bars  : $SHOW_ERRORBARS"
echo "  tc kind   : $TC_DIRNAME"
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
