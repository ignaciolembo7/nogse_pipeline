#!/usr/bin/env bash

set -u -o pipefail

# Driver script for the brain DWI signal-extraction batch.
# All workflow-specific paths are defined here, while logging and case
# execution helpers come from `coreg_batch_lib.sh`.

SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_HOME/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"
COREG_SCRIPT="$REPO_ROOT/src/signal_extraction/coreg_extract.py"
SUBJECTS_DIR="$PROJECT_ROOT/Data-signals/DATA_PROCESSED/subjects"

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
OUT_ROOT="$PROJECT_ROOT/Data-signals"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/brains}"
REQUIRE_SUBJECTS_DIR=1
# Which dcm2niix conflict variant to use: "none", "a", "b", ... or "all".
DWI_VARIANT="none"
# Set to 1 only for acquisitions that must be collapsed into one mean image
# and one signal row per sequence.
USE_MEAN="0"
# Number of initial volumes to discard when USE_MEAN="1".
DUMMY_SCANS="0"
# Set to 1 to reuse existing reference images, or 0 to overwrite them.
REUSE_REFERENCE="0"
# ------------------------------------------------------------------
# ------------------------------------------------------------------

case "$REUSE_REFERENCE" in
  0|1)
    ;;
  *)
    echo "ERROR: REUSE_REFERENCE must be 0 or 1, got: $REUSE_REFERENCE"
    exit 1
    ;;
esac
export REUSE_REFERENCE

# Load shared batch helpers after defining all required variables.
source "$SCRIPT_HOME/../helpers/coreg_batch_lib.sh"

init_run "run_BRAINS-denoised_topup_signal_extraction"

run_case "20220622_BRAIN" \
  --exp-root "$PROJECT_ROOT/Data-NIFTI-BRAINS-denoised_topup/20220622_BRAIN" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "$OUT_ROOT" \
  --dwi-variant "$DWI_VARIANT" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle

run_case "20230619_BRAIN-3" \
  --exp-root "$PROJECT_ROOT/Data-NIFTI-BRAINS-denoised_topup/20230619_BRAIN-3" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "$OUT_ROOT" \
  --dwi-variant "$DWI_VARIANT" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle \
  --syringe-mask-t1 "syringe_mask.nii.gz"

run_case "20230623_BRAIN-4" \
  --exp-root "$PROJECT_ROOT/Data-NIFTI-BRAINS-denoised_topup/20230623_BRAIN-4" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "$OUT_ROOT" \
  --dwi-variant "$DWI_VARIANT" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle \
  --syringe-mask-t1 "syringe_mask.nii.gz"

run_case "20230623_LUDG-2" \
  --exp-root "$PROJECT_ROOT/Data-NIFTI-BRAINS-denoised_topup/20230623_LUDG-2" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "$OUT_ROOT" \
  --dwi-variant "$DWI_VARIANT" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle \
  --syringe-mask-t1 "syringe_mask.nii.gz"

run_case "20230629_MBBL-2" \
  --exp-root "$PROJECT_ROOT/Data-NIFTI-BRAINS-denoised_topup/20230629_MBBL-2" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "$OUT_ROOT" \
  --dwi-variant "$DWI_VARIANT" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle \
  --syringe-mask-t1 "syringe_mask.nii.gz"

run_case "20230710_LUDG-3" \
  --exp-root "$PROJECT_ROOT/Data-NIFTI-BRAINS-denoised_topup/20230710_LUDG-3" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "$OUT_ROOT" \
  --dwi-variant "$DWI_VARIANT" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle \
  --syringe-mask-dwi "syringe_mask.nii.gz"

run_case "20230630_MBBL-3" \
  --exp-root "$PROJECT_ROOT/Data-NIFTI-BRAINS-denoised_topup/20230630_MBBL-3" \
  --subjects-dir "$SUBJECTS_DIR" \
  --out-root "$OUT_ROOT" \
  --dwi-variant "$DWI_VARIANT" \
  --cut-token "_hifi_images_den" \
  --atlas-roi 4:Left-Lateral-Ventricle \
  --atlas-roi 43:Right-Lateral-Ventricle \
  --syringe-mask-t1 "syringe_mask.nii.gz"

finish_run
exit $?
