#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/helpers/master_table_common.sh"
pipeline_maybe_step_help fit_global_signal "$@"
pipeline_setup_common
pipeline_set_dataset_defaults "${TYPE_SUBJ:-${DATASET:?TYPE_SUBJ or DATASET is required}}"

FIT_GLOBAL_SIGNAL_SCRIPT="${FIT_GLOBAL_SIGNAL_SCRIPT:-$REPO_ROOT/scripts/fitting/fit_global_signal.py}"
GLOBAL_SIGNAL_OUT_ROOT="${GLOBAL_SIGNAL_OUT_ROOT:-$ANALYSIS_ROOT/fits/${TYPE_SEQ}_signal_mixed_global_master}"

if [[ "$TYPE_SEQ" == "nogse" ]]; then
    GLOBAL_SIGNAL_MODEL_DEFAULT="${GLOBAL_SIGNAL_MODEL_DEFAULT:-nogse_mixed_offset}"
    GLOBAL_SIGNAL_G_TYPE="${GLOBAL_SIGNAL_G_TYPE:-g}"
    GLOBAL_SIGNAL_DIRECTIONS="${GLOBAL_SIGNAL_DIRECTIONS:-ALL}"
    if [[ "$DATASET" == "brains" ]]; then
        GLOBAL_SIGNAL_CORR_ROI="${GLOBAL_SIGNAL_CORR_ROI:-Syringe}"
        GLOBAL_SIGNAL_D0_FIXED="${GLOBAL_SIGNAL_D0_FIXED:-3.2e-12}"
    else
        GLOBAL_SIGNAL_CORR_ROI="${GLOBAL_SIGNAL_CORR_ROI:-water}"
        GLOBAL_SIGNAL_D0_FIXED="${GLOBAL_SIGNAL_D0_FIXED:-2.3e-12}"
    fi
elif [[ "$DATASET" == "brains" ]]; then
    GLOBAL_SIGNAL_MODEL_DEFAULT="${GLOBAL_SIGNAL_MODEL_DEFAULT:-ogse_mixed_offset}"
    GLOBAL_SIGNAL_G_TYPE="${GLOBAL_SIGNAL_G_TYPE:-g_thorsten}"
    GLOBAL_SIGNAL_DIRECTIONS="${GLOBAL_SIGNAL_DIRECTIONS:-long tra}"
    GLOBAL_SIGNAL_CORR_ROI="${GLOBAL_SIGNAL_CORR_ROI:-Syringe}"
    GLOBAL_SIGNAL_D0_FIXED="${GLOBAL_SIGNAL_D0_FIXED:-3.2e-12}"
else
    GLOBAL_SIGNAL_MODEL_DEFAULT="${GLOBAL_SIGNAL_MODEL_DEFAULT:-ogse_mixed_offset}"
    GLOBAL_SIGNAL_G_TYPE="${GLOBAL_SIGNAL_G_TYPE:-g}"
    GLOBAL_SIGNAL_DIRECTIONS="${GLOBAL_SIGNAL_DIRECTIONS:-ALL}"
    GLOBAL_SIGNAL_CORR_ROI="${GLOBAL_SIGNAL_CORR_ROI:-water}"
    GLOBAL_SIGNAL_D0_FIXED="${GLOBAL_SIGNAL_D0_FIXED:-2.3e-12}"
fi

pipeline_require_file "$FIT_GLOBAL_SIGNAL_SCRIPT" "global signal fit script"
pipeline_require_file "$MASTER_PARQUET" "master table"
mkdir -p "$GLOBAL_SIGNAL_OUT_ROOT"

args=(
    --master-parquet "$MASTER_PARQUET"
    --row-kind "${GLOBAL_SIGNAL_ROW_KIND:-signal_rotated}"
    --out_root "$GLOBAL_SIGNAL_OUT_ROOT"
    --type-seq "$TYPE_SEQ"
    --model "${GLOBAL_SIGNAL_MODEL:-$GLOBAL_SIGNAL_MODEL_DEFAULT}"
    --ycol "${GLOBAL_SIGNAL_YCOL:-value}"
    --g_type "$GLOBAL_SIGNAL_G_TYPE"
    --stat "${GLOBAL_SIGNAL_STAT:-avg}"
    --min_points "${GLOBAL_SIGNAL_MIN_POINTS:-4}"
    --tc_mode "${GLOBAL_SIGNAL_TC_MODE:-global_td}"
    --alpha_mode "${GLOBAL_SIGNAL_ALPHA_MODE:-global_td}"
    --RN_mode "${GLOBAL_SIGNAL_RN_MODE:-global_td}"
    --M0_mode "${GLOBAL_SIGNAL_M0_MODE:-global_contrast}"
    --C_mode "${GLOBAL_SIGNAL_C_MODE:-global_contrast}"
    --D0_mode "${GLOBAL_SIGNAL_D0_MODE:-fixed}"
    --D0_fixed "$GLOBAL_SIGNAL_D0_FIXED"
)

if [[ "$GLOBAL_SIGNAL_DIRECTIONS" != "ALL" ]]; then
    read -r -a direction_args <<< "${GLOBAL_SIGNAL_DIRECTIONS//,/ }"
    args+=(--directions "${direction_args[@]}")
fi
if [[ "${GLOBAL_SIGNAL_ROIS:-ALL}" != "ALL" ]]; then
    read -r -a roi_args <<< "${GLOBAL_SIGNAL_ROIS//,/ }"
    args+=(--rois "${roi_args[@]}")
fi
if [[ "${GLOBAL_SIGNAL_SUBJS:-ALL}" != "ALL" ]]; then
    read -r -a subj_args <<< "${GLOBAL_SIGNAL_SUBJS//,/ }"
    args+=(--subjs "${subj_args[@]}")
fi

if [[ "${GLOBAL_SIGNAL_APPLY_GRAD_CORR:-false}" == "true" ]]; then
    GLOBAL_SIGNAL_CORR_XLSX="${GLOBAL_SIGNAL_CORR_XLSX:-$ANALYSIS_ROOT/fits/grad_correction_master/${GLOBAL_SIGNAL_CORR_ROI}.grad_correction.xlsx}"
    pipeline_require_file "$GLOBAL_SIGNAL_CORR_XLSX" "gradient correction table"
    args+=(--apply_grad_corr --corr_xlsx "$GLOBAL_SIGNAL_CORR_XLSX" --corr_roi "$GLOBAL_SIGNAL_CORR_ROI")
else
    args+=(--no_grad_corr)
fi

"$PY" "$FIT_GLOBAL_SIGNAL_SCRIPT" "${args[@]}" ${GLOBAL_SIGNAL_EXTRA_ARGS:-}
