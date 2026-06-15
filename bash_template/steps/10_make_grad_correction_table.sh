#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/helpers/master_table_common.sh"
pipeline_maybe_step_help grad_correction "$@"
pipeline_setup_common
pipeline_set_dataset_defaults "${TYPE_SUBJ:-${DATASET:?TYPE_SUBJ or DATASET is required}}"

GRAD_CORR_SCRIPT="${GRAD_CORR_SCRIPT:-$REPO_ROOT/scripts/data/make_grad_correction_table.py}"
SIGNAL_FITS_ROOT="${SIGNAL_FITS_ROOT:-$ANALYSIS_ROOT/fits/${TYPE_SEQ}_signal_master}"
CONTRAST_FITS_ROOT="${CONTRAST_FITS_ROOT:-$ANALYSIS_ROOT/fits/${TYPE_SEQ}_contrast_master}"
CONTRAST_DATA_ROOT="${CONTRAST_DATA_ROOT:-$ANALYSIS_ROOT/contrast-data-master}"
GRAD_CORR_OUT_DIR="${GRAD_CORR_OUT_DIR:-$ANALYSIS_ROOT/fits/grad_correction_master}"

if [[ "$DATASET" == "brains" ]]; then
    GRAD_CORR_ROI="${GRAD_CORR_ROI:-Syringe}"
else
    GRAD_CORR_ROI="${GRAD_CORR_ROI:-water}"
fi

pipeline_require_file "$GRAD_CORR_SCRIPT" "gradient correction script"
mkdir -p "$GRAD_CORR_OUT_DIR"

"$PY" "$GRAD_CORR_SCRIPT" \
    --roi "$GRAD_CORR_ROI" \
    --exp-fits-root "$SIGNAL_FITS_ROOT" \
    --nogse-root "$CONTRAST_FITS_ROOT" \
    --contrast-data-root "$CONTRAST_DATA_ROOT" \
    --out-xlsx "$GRAD_CORR_OUT_DIR/${GRAD_CORR_ROI}.grad_correction.xlsx" \
    --out-csv "$GRAD_CORR_OUT_DIR/${GRAD_CORR_ROI}.grad_correction.csv" \
    ${GRAD_CORR_EXTRA_ARGS:-}
