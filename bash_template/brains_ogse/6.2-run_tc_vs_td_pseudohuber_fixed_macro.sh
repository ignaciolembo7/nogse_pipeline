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
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$PROJECT_ROOT/analysis/brains/ogse_experiments}"
CONTRAST_SOURCE="${CONTRAST_SOURCE:-fitted_resampled}" # direct or fitted_resampled
SIGNAL_MODEL="${SIGNAL_MODEL:-rest_offset_globC}"
SIGNAL_G_TYPE="${SIGNAL_G_TYPE:-g}"
FIT_CORR="${FIT_CORR:-${USE_CORR:-true}}"
SUMMARY_ALPHA="$ANALYSIS_ROOT/alpha_macro/N1/summary_alpha_values.xlsx"
YCOL="${YCOL:-}"
# tc_peak or tc_fit; inferred from YCOL when omitted
TC_ANALYSIS="${TC_ANALYSIS:-tc_fit}" 
PEAK_D0_FIX="${PEAK_D0_FIX:-3.2e-12}"
PEAK_GAMMA="${PEAK_GAMMA:-267.5221900}"
EXCLUDE_TD_MS="${EXCLUDE_TD_MS:-76}"
SHOW_ERRORBARS="1"
ROIS="${ROIS:-AntCC,MidAntCC,CentralCC,MidPostCC,PostCC}"
TD_MIN_MS="75"
TD_MAX_MS="225"
C_FIXED="FREE"
C_MIN="0"
C_MAX="10"
DELTA_FIXED="FREE"
DELTA_MIN="1e-6"
DELTA_MAX="10000"
EXCLUDE_MATCHES=()
# ------------------------------------------------------------------
# ------------------------------------------------------------------
while (( $# > 0 )); do
    case "$1" in
        --corr)
            FIT_CORR=true
            shift
            ;;
        --no-corr|--sin-corr|--uncorr)
            FIT_CORR=false
            shift
            ;;
        *)
            echo "ERROR: Unknown argument for $0: $1" >&2
            echo "Use --corr or --no-corr. Other settings are controlled with environment variables." >&2
            exit 1
            ;;
    esac
done
if [[ "$CONTRAST_SOURCE" != "direct" && "$CONTRAST_SOURCE" != "fitted_resampled" ]]; then
    echo "ERROR: CONTRAST_SOURCE must be 'direct' or 'fitted_resampled'. Got: $CONTRAST_SOURCE" >&2
    exit 1
fi
if [[ -n "${TC_ANALYSIS// }" && "$TC_ANALYSIS" != "tc_peak" && "$TC_ANALYSIS" != "tc_fit" ]]; then
    echo "ERROR: TC_ANALYSIS must be empty, 'tc_peak', or 'tc_fit'. Got: $TC_ANALYSIS" >&2
    exit 1
fi
if [[ -z "${FITS_ROOT_SUFFIX+x}" ]]; then
    if [[ "$SIGNAL_MODEL" == "mixed_global" ]]; then
        FITS_ROOT_SUFFIX=""
    else
        case "${FIT_CORR,,}" in
            1|true|yes|y|corr)
                FITS_ROOT_SUFFIX="_corr"
                ;;
            0|false|no|n|none|uncorr|no-corr|sin-corr)
                FITS_ROOT_SUFFIX=""
                ;;
            *)
                echo "ERROR: FIT_CORR must be true/false. Got: $FIT_CORR" >&2
                exit 1
                ;;
        esac
    fi
fi
if [[ "$CONTRAST_SOURCE" == "fitted_resampled" ]]; then
    FIT_ROOT="${FIT_ROOT:-$ANALYSIS_ROOT/fits/ogse_contrast_vs_gresampled_${SIGNAL_MODEL}${FITS_ROOT_SUFFIX}}"
    CONTRAST_ROOT="${CONTRAST_ROOT:-$FIT_ROOT/contrast}"
else
    FIT_ROOT="${FIT_ROOT:-$ANALYSIS_ROOT/fits/ogse_contrast_vs_g_${SIGNAL_MODEL}${FITS_ROOT_SUFFIX}}"
    CONTRAST_ROOT="${CONTRAST_ROOT:-}"
fi
if [[ -z "${TC_ANALYSIS// }" ]]; then
    case "$YCOL" in
        tc_fit_ms|tc_ms)
            TC_ANALYSIS="tc_fit"
            ;;
        *)
            TC_ANALYSIS="tc_peak"
            ;;
    esac
fi
if [[ -z "${YCOL// }" ]]; then
    if [[ "$TC_ANALYSIS" == "tc_fit" ]]; then
        YCOL="tc_fit_ms"
    elif [[ "$CONTRAST_SOURCE" == "fitted_resampled" ]]; then
        YCOL="tc_peak_resampled_data_ms"
    else
        YCOL="tc_peak_ms"
    fi
fi
GROUPFITS_TAG=""
if [[ "$TC_ANALYSIS" == "tc_fit" ]]; then
    GROUPFITS_TAG="_tcfit"
fi
GROUPFITS="${GROUPFITS:-$FIT_ROOT/groupfits_rest${GROUPFITS_TAG}.parquet}"

case "$YCOL" in
    tc_peak_ms)
        TC_DIRNAME="tcpeak_vs_td"
        ;;
    tc_peak_resampled_ms)
        TC_DIRNAME="tcpeak_resampled_vs_td"
        ;;
    tc_peak_resampled_data_ms)
        TC_DIRNAME="tcpeak_resampled_data_vs_td"
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
if [[ "$ROIS" != "ALL" ]]; then
    read -r -a roi_list <<< "${ROIS//,/ }"
    if (( ${#roi_list[@]} > 0 )); then
        extra_args+=(--rois "${roi_list[@]}")
    fi
fi
if [[ "$YCOL" == "tc_peak_resampled_data_ms" ]]; then
    if [[ -z "${CONTRAST_ROOT// }" ]]; then
        echo "ERROR: YCOL=tc_peak_resampled_data_ms requires CONTRAST_ROOT." >&2
        exit 1
    fi
    extra_args+=(--add-resampled-data-peaks --contrast-root "$CONTRAST_ROOT" --peak-D0-fix "$PEAK_D0_FIX" --peak-gamma "$PEAK_GAMMA")
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
echo "Contrast source: $CONTRAST_SOURCE"
echo "TC analysis   : $TC_ANALYSIS"
echo "Signal model  : $SIGNAL_MODEL"
echo "Fit corr      : $FIT_CORR"
echo "Fits suffix   : ${FITS_ROOT_SUFFIX:-<none>}"
echo "ROIs           : $ROIS"
echo "Method        : $METHOD"
echo "Groupfits     : $GROUPFITS"
echo "Contrast root : ${CONTRAST_ROOT:-<none>}"
echo "Summary alpha : $SUMMARY_ALPHA"
echo "Y column      : $YCOL"
echo "Peak D0/gamma : $PEAK_D0_FIX / $PEAK_GAMMA"
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
