#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DEFAULT_PY="python"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    DEFAULT_PY="${CONDA_PREFIX}/bin/python"
elif [[ -x "/home/ignacio.lemboferrari@unitn.it/.conda/envs/nogse_pipe_env/bin/python" ]]; then
    DEFAULT_PY="/home/ignacio.lemboferrari@unitn.it/.conda/envs/nogse_pipe_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    DEFAULT_PY="$(command -v python3)"
fi
PY="${PY:-$DEFAULT_PY}"
TC_SCRIPT="$REPO_ROOT/scripts/run_tc_vs_td.py"

METHOD="${METHOD:-pseudohuber_fixed_macro}"
GROUPFITS="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/fit_rest_ogse_contrast_corr/groupfits_rest.parquet"
SUMMARY_ALPHA="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/alpha_macro/N1/summary_alpha_values.xlsx"
YCOL="tc_peak_ms"
EXCLUDE_TD_MS="75.1,209.1"
SHOW_ERRORBARS="0"
TD_MIN_MS="0"
TD_MAX_MS="250"
C_FIXED="FREE"
C_MIN="0"
C_MAX="INF"
DELTA_FIXED="FREE"
# To force delta to 0 ms, uncomment the next line.
# DELTA_FIXED="0"
DELTA_MIN="1e-6"
DELTA_MAX="250"
EXCLUDE_MATCHES=()
if [[ "$YCOL" == "tc_peak_ms" ]]; then
    TC_DIRNAME="tcpeak_vs_td"
elif [[ "$YCOL" == "tc_fit_ms" || "$YCOL" == "tc_ms" ]]; then
    TC_DIRNAME="tcfit_vs_td"
else
    TC_DIRNAME="${YCOL}_vs_td"
fi
OUT_DIR="$PROJECT_ROOT/analysis/phantoms/ogse_experiments/fits/fit_rest_ogse_contrast_corr/$TC_DIRNAME/$METHOD" #/$YCOL"

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
echo "Dataset       : phantoms"
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

# To release alpha_macro (fit it instead of using fixed macro values), run:
# METHOD="pseudohuber_free" bash "$0"
