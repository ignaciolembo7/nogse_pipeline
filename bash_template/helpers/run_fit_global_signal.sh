#!/usr/bin/env bash
set -euo pipefail

HELPER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_ROOT="$(cd "$HELPER_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$TEMPLATE_ROOT/../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

# ------------------------------------------------------------------
# Fit one signal model jointly across td/N curves.
#
# Each parameter can be fixed, free per curve, global across the subj/roi/direction
# group, or shared inside each N1/N2 contrast set.
# ------------------------------------------------------------------

DEFAULT_PY="python"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    DEFAULT_PY="${CONDA_PREFIX}/bin/python"
elif [[ -x "$HOME/.conda/envs/nogse_pipe_env/bin/python" ]]; then
    DEFAULT_PY="$HOME/.conda/envs/nogse_pipe_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    DEFAULT_PY="$(command -v python3)"
fi
PY="${PY:-$DEFAULT_PY}"

FAMILY="${FAMILY:-ogse}"               # ogse or nogse.
DATASET_KIND="${DATASET_KIND:-brains}" # brains or phantoms.
case "${FAMILY,,}" in
    ogse|nogse) ;;
    *)
        echo "ERROR: FAMILY must be ogse or nogse, got: $FAMILY" >&2
        exit 1
        ;;
esac

case "${DATASET_KIND,,}" in
    brains)
        DATASET_LABEL="Brains"
        ANALYSIS_ROOT_DEFAULT="$PROJECT_ROOT/analysis/brains/${FAMILY,,}_experiments"
        if [[ "${FAMILY,,}" == "ogse" ]]; then
            DATA_ROOT_DEFAULT="$ANALYSIS_ROOT_DEFAULT/data-rotated/tables"
            CONTRAST_ROOT_DEFAULT="$ANALYSIS_ROOT_DEFAULT/contrast-data-rotated"
            CORR_XLSX_DEFAULT="$ANALYSIS_ROOT_DEFAULT/fits/grad_correction_rotated/Syringe.grad_correction_rotated.xlsx"
        else
            DATA_ROOT_DEFAULT="$ANALYSIS_ROOT_DEFAULT/data/tables"
            CONTRAST_ROOT_DEFAULT="$ANALYSIS_ROOT_DEFAULT/contrast-data"
            CORR_XLSX_DEFAULT="$ANALYSIS_ROOT_DEFAULT/fits/grad_correction/Syringe.grad_correction.xlsx"
        fi
        CORR_ROI_DEFAULT="Syringe"
        G_TYPE_DEFAULT="g_thorsten"
        DIRECTIONS_DEFAULT="long tra"
        ROIS_DEFAULT="ALL"
        D0_FIXED_DEFAULT="3.2e-12"
        M0_BOUNDS_DEFAULT="0.0 10000.0"
        C_BOUNDS_DEFAULT="-10000.0 10000.0"
        RN_BOUNDS_DEFAULT="0.0 1000.0"
        D0_BOUNDS_DEFAULT="1e-13 1e-10"
        ;;
    phantoms)
        DATASET_LABEL="Phantoms"
        ANALYSIS_ROOT_DEFAULT="$PROJECT_ROOT/analysis/phantoms/${FAMILY,,}_experiments"
        if [[ "${FAMILY,,}" == "ogse" ]]; then
            DATA_ROOT_DEFAULT="$ANALYSIS_ROOT_DEFAULT/data"
        else
            DATA_ROOT_DEFAULT="$ANALYSIS_ROOT_DEFAULT/data/tables"
        fi
        CONTRAST_ROOT_DEFAULT="$ANALYSIS_ROOT_DEFAULT/contrast-data"
        CORR_XLSX_DEFAULT="$ANALYSIS_ROOT_DEFAULT/fits/grad_correction/water.grad_correction.xlsx"
        CORR_ROI_DEFAULT="water"
        G_TYPE_DEFAULT="g"
        DIRECTIONS_DEFAULT="ALL"
        ROIS_DEFAULT="ALL"
        D0_FIXED_DEFAULT="2.3e-12"
        M0_BOUNDS_DEFAULT="0.0 1000000000.0"
        C_BOUNDS_DEFAULT="-1000000000.0 1000000000.0"
        RN_BOUNDS_DEFAULT="0.0 1000000000.0"
        D0_BOUNDS_DEFAULT="1e-16 1e-10"
        ;;
    *)
        echo "ERROR: DATASET_KIND must be brains or phantoms, got: $DATASET_KIND" >&2
        exit 1
        ;;
esac

FIT_SCRIPT="${FIT_SCRIPT:-$REPO_ROOT/scripts/fit_global_signal.py}"
EXPORT_RESAMPLED_SCRIPT="${EXPORT_RESAMPLED_SCRIPT:-$REPO_ROOT/scripts/export_ogse_resampled_contrasts_from_fits.py}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-$ANALYSIS_ROOT_DEFAULT}"
DATA_ROOT="${DATA_ROOT:-$DATA_ROOT_DEFAULT}"
CONTRAST_ROOT="${CONTRAST_ROOT:-$CONTRAST_ROOT_DEFAULT}"
APPLY_GRAD_CORR="${APPLY_GRAD_CORR:-true}"
CORR_XLSX="${CORR_XLSX:-$CORR_XLSX_DEFAULT}"
CORR_ROI="${CORR_ROI:-$CORR_ROI_DEFAULT}"
CORR_TOL_MS="${CORR_TOL_MS:-1e-3}"
CORR_SHEET="${CORR_SHEET:-}"
CORR_MISSING="${CORR_MISSING:-error}" # error, identity, or skip.

MODEL="${MODEL:-${FAMILY,,}_mixed_offset}"
ROOT_SUFFIX=""
if [[ "${APPLY_GRAD_CORR,,}" == "true" ]]; then
    ROOT_SUFFIX="_corr"
fi
OUT_ROOT="${OUT_ROOT:-$ANALYSIS_ROOT/fits/${FAMILY,,}_signal_vs_g_${MODEL}_global${ROOT_SUFFIX}}"
CONTRAST_OUT_DIR="${CONTRAST_OUT_DIR:-$OUT_ROOT/contrast}"
if [[ "${FAMILY,,}" == "ogse" ]]; then
    EXPORT_RESAMPLED_CONTRASTS="${EXPORT_RESAMPLED_CONTRASTS:-true}"
else
    EXPORT_RESAMPLED_CONTRASTS="${EXPORT_RESAMPLED_CONTRASTS:-false}"
fi
WRITE_CSV="${WRITE_CSV:-false}"
WRITE_SIGNAL_TABLES="${WRITE_SIGNAL_TABLES:-false}"

FILE_PATTERN="${FILE_PATTERN:-*.long.parquet}"
YCOL="${YCOL:-value}"
G_TYPE="${G_TYPE:-$G_TYPE_DEFAULT}"
STAT="${STAT:-avg}"
ROIS="${ROIS:-$ROIS_DEFAULT}"
DIRECTIONS="${DIRECTIONS:-$DIRECTIONS_DEFAULT}"
N_FIT="${N_FIT:-}"
MIN_POINTS="${MIN_POINTS:-4}"

TC_MODE="${TC_MODE:-global_td}"
ALPHA_MODE="${ALPHA_MODE:-global_td}"
RN_MODE="${RN_MODE:-global_td}"
M0_MODE="${M0_MODE:-global_contrast}"
C_MODE="${C_MODE:-global_contrast}"
D0_MODE="${D0_MODE:-fixed}"
D0_FIXED="${D0_FIXED:-$D0_FIXED_DEFAULT}"
TC_INIT="${TC_INIT:-5.0}"
TC_FIXED="${TC_FIXED:-}"
ALPHA_INIT="${ALPHA_INIT:-0.5}"
ALPHA_FIXED="${ALPHA_FIXED:-}"
M0_INIT="${M0_INIT:-}"
M0_FIXED="${M0_FIXED:-}"
C_INIT="${C_INIT:-0.0}"
C_FIXED="${C_FIXED:-}"
RN_INIT="${RN_INIT:-0.0}"
RN_FIXED="${RN_FIXED:-10}"

TC_BOUNDS="${TC_BOUNDS:-0.1 1000.0}"
ALPHA_BOUNDS="${ALPHA_BOUNDS:-0.0 1.0}"
M0_BOUNDS="${M0_BOUNDS:-$M0_BOUNDS_DEFAULT}"
C_BOUNDS="${C_BOUNDS:-$C_BOUNDS_DEFAULT}"
RN_BOUNDS="${RN_BOUNDS:-$RN_BOUNDS_DEFAULT}"
D0_BOUNDS="${D0_BOUNDS:-$D0_BOUNDS_DEFAULT}"
MAX_NFEV="${MAX_NFEV:-400000}"
RESAMPLED_GRID_MIN_MTM="${RESAMPLED_GRID_MIN_MTM:-0}"
RESAMPLED_GRID_MAX_MTM="${RESAMPLED_GRID_MAX_MTM:-90}"
RESAMPLED_GRID_N="${RESAMPLED_GRID_N:-1000}"
RESAMPLED_GRID_MAX_MODE="${RESAMPLED_GRID_MAX_MODE:-observed_pair_max}"
RESAMPLED_CONTRAST_MODE="${RESAMPLED_CONTRAST_MODE:-signal_fit}"
EXPORT_FIT_PARAMS_PATTERN="${EXPORT_FIT_PARAMS_PATTERN:-fit_params.${MODEL}.*.parquet}"
PEAK_D0_FIX="${PEAK_D0_FIX:-3.2e-12}"
PEAK_GAMMA="${PEAK_GAMMA:-267.5221900}"

if [[ ! -f "$FIT_SCRIPT" ]]; then
    echo "ERROR: fit script not found: $FIT_SCRIPT" >&2
    exit 1
fi

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "ERROR: data root not found: $DATA_ROOT" >&2
    exit 1
fi

mkdir -p "$OUT_ROOT"

contrast_args=()
if [[ -d "$CONTRAST_ROOT" ]]; then
    contrast_args+=(--contrast_root "$CONTRAST_ROOT")
elif [[ -n "${CONTRAST_ROOT+x}" && "$CONTRAST_ROOT" != "$CONTRAST_ROOT_DEFAULT" ]]; then
    echo "ERROR: contrast root not found: $CONTRAST_ROOT" >&2
    exit 1
fi

read -r -a direction_args <<< "${DIRECTIONS//,/ }"
read -r -a roi_args <<< "${ROIS//,/ }"
read -r -a tc_bounds_args <<< "${TC_BOUNDS//,/ }"
read -r -a alpha_bounds_args <<< "${ALPHA_BOUNDS//,/ }"
read -r -a m0_bounds_args <<< "${M0_BOUNDS//,/ }"
read -r -a c_bounds_args <<< "${C_BOUNDS//,/ }"
read -r -a rn_bounds_args <<< "${RN_BOUNDS//,/ }"
read -r -a d0_bounds_args <<< "${D0_BOUNDS//,/ }"

extra_args=()
if [[ "${APPLY_GRAD_CORR,,}" == "true" ]]; then
    extra_args+=(--apply_grad_corr --corr_xlsx "$CORR_XLSX" --corr_roi "$CORR_ROI" --corr_tol_ms "$CORR_TOL_MS" --corr_missing "$CORR_MISSING")
    if [[ -n "${CORR_SHEET// }" ]]; then
        extra_args+=(--corr_sheet "$CORR_SHEET")
    fi
else
    extra_args+=(--no_grad_corr)
fi
if [[ -n "${N_FIT// }" ]]; then
    extra_args+=(--n_fit "$N_FIT")
fi
if [[ -n "${TC_FIXED// }" ]]; then
    extra_args+=(--tc_fixed "$TC_FIXED")
fi
if [[ -n "${ALPHA_FIXED// }" ]]; then
    extra_args+=(--alpha_fixed "$ALPHA_FIXED")
fi
if [[ -n "${M0_INIT// }" ]]; then
    extra_args+=(--M0_init "$M0_INIT")
fi
if [[ -n "${M0_FIXED// }" ]]; then
    extra_args+=(--M0_fixed "$M0_FIXED")
fi
if [[ -n "${C_FIXED// }" ]]; then
    extra_args+=(--C_fixed "$C_FIXED")
fi
if [[ -n "${RN_FIXED// }" ]]; then
    extra_args+=(--RN_fixed "$RN_FIXED")
fi
if [[ "${WRITE_CSV,,}" == "true" ]]; then
    extra_args+=(--write_csv)
fi
if [[ "${WRITE_SIGNAL_TABLES,,}" == "true" ]]; then
    extra_args+=(--write_signal_tables)
fi
if [[ "$DIRECTIONS" != "ALL" ]]; then
    extra_args+=(--directions "${direction_args[@]}")
fi
if [[ "$ROIS" != "ALL" ]]; then
    extra_args+=(--rois "${roi_args[@]}")
fi

declare -a files=()
while read -r path; do
    [[ -z "$path" ]] && continue
    base="$(basename "$path")"
    if [[ "$base" == *.Dproj.long.parquet || "$base" == *.monoexp.Dproj.long.parquet ]]; then
        continue
    fi
    if [[ "$base" == *"_fitresamp-"* ]]; then
        continue
    fi
    files+=("$path")
done < <(find "$DATA_ROOT" -type f -name "$FILE_PATTERN" | sort)

if (( ${#files[@]} == 0 )); then
    echo "ERROR: no input tables matched FILE_PATTERN=$FILE_PATTERN under DATA_ROOT=$DATA_ROOT" >&2
    exit 1
fi

echo "$DATASET_LABEL ${FAMILY^^} global signal fit"
echo "  Model        : $MODEL"
echo "  Data root    : $DATA_ROOT"
echo "  Files        : ${#files[@]}"
echo "  Out root     : $OUT_ROOT"
echo "  y column     : $YCOL"
echo "  G column     : $G_TYPE"
echo "  stat         : $STAT"
echo "  ROIs         : $ROIS"
echo "  Directions   : $DIRECTIONS"
echo "  Grad corr    : $APPLY_GRAD_CORR"
echo "  Corr missing : $CORR_MISSING"
if (( ${#contrast_args[@]} > 0 )); then
    echo "  Contrast root: $CONTRAST_ROOT"
else
    echo "  Contrast root: not used"
fi
echo "  Modes        : tc=$TC_MODE alpha=$ALPHA_MODE RN=$RN_MODE M0=$M0_MODE C=$C_MODE D0=$D0_MODE"
echo "  D0 mode      : $D0_MODE (value/seed=$D0_FIXED m^2/ms)"
echo "  Write CSV    : $WRITE_CSV"
echo "  Signal tables: $WRITE_SIGNAL_TABLES"

"$PY" "$FIT_SCRIPT" \
    "${files[@]}" \
    --out_root "$OUT_ROOT" \
    --family "${FAMILY,,}" \
    --model "$MODEL" \
    "${contrast_args[@]}" \
    --ycol "$YCOL" \
    --g_type "$G_TYPE" \
    --stat "$STAT" \
    --min_points "$MIN_POINTS" \
    --tc_mode "$TC_MODE" \
    --alpha_mode "$ALPHA_MODE" \
    --RN_mode "$RN_MODE" \
    --M0_mode "$M0_MODE" \
    --C_mode "$C_MODE" \
    --D0_mode "$D0_MODE" \
    --D0_fixed "$D0_FIXED" \
    --tc_init "$TC_INIT" \
    --alpha_init "$ALPHA_INIT" \
    --C_init "$C_INIT" \
    --RN_init "$RN_INIT" \
    --tc_bounds "${tc_bounds_args[@]}" \
    --alpha_bounds "${alpha_bounds_args[@]}" \
    --M0_bounds "${m0_bounds_args[@]}" \
    --C_bounds "${c_bounds_args[@]}" \
    --RN_bounds "${rn_bounds_args[@]}" \
    --D0_bounds "${d0_bounds_args[@]}" \
    --max_nfev "$MAX_NFEV" \
    "${extra_args[@]}"

if [[ "${EXPORT_RESAMPLED_CONTRASTS,,}" == "true" ]]; then
    if [[ "${FAMILY,,}" != "ogse" ]]; then
        echo "ERROR: resampled contrast export is only implemented for OGSE global signal fits." >&2
        exit 1
    fi
    echo
    echo "Exporting resampled contrast outputs"
    echo "  Out dir      : $CONTRAST_OUT_DIR"
    "$PY" "$EXPORT_RESAMPLED_SCRIPT" \
        "$OUT_ROOT" \
        --contrast-root "$CONTRAST_ROOT" \
        --out-dir "$CONTRAST_OUT_DIR" \
        --pattern "$EXPORT_FIT_PARAMS_PATTERN" \
        --grid-min-mTm "$RESAMPLED_GRID_MIN_MTM" \
        --grid-max-mTm "$RESAMPLED_GRID_MAX_MTM" \
        --grid-max-mode "$RESAMPLED_GRID_MAX_MODE" \
        --grid-n "$RESAMPLED_GRID_N" \
        --contrast-mode "$RESAMPLED_CONTRAST_MODE" \
        --models "$MODEL" \
        --peak-D0-fix "$PEAK_D0_FIX" \
        --peak-gamma "$PEAK_GAMMA"
fi
