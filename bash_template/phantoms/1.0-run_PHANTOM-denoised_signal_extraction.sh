#!/usr/bin/env bash

set -u -o pipefail

# Driver script for the phantom DWI signal-extraction batch.
# All workflow-specific paths are defined here, while logging and case
# execution helpers come from `coreg_batch_lib.sh`.

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_HOME/../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"
COREG_SCRIPT="$REPO_ROOT/src/signal_extraction/coreg_extract.py"
OUT_ROOT="$PROJECT_ROOT/Data-signals"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/phantoms}"
EXP_ROOT="$PROJECT_ROOT/Data-NIFTI/20260122-PHANTOM_FIBER/QUALITY_JACK_19800122TMSF"
OUT_SUBJ_REL="20260122-PHANTOM_FIBER/QUALITY_JACK_19800122TMSF"
CUT_TOKEN=""
# Which dcm2niix conflict variant to use: "none", "a", "b", ... or "all".
DWI_VARIANT="none"
# Set to 1 only for acquisitions that must be collapsed into one mean image
# and one signal row per sequence.
USE_MEAN="1"
# Number of initial volumes to discard when USE_MEAN="1".
DUMMY_SCANS="5"
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

# Load shared batch helpers after defining all required variables.
source "$SCRIPT_HOME/../helpers/coreg_batch_lib.sh"

init_run "run_PHANTOM-denoised_signal_extraction"

EXTRA_ARGS=()
if [[ "$USE_MEAN" == "1" ]]; then
  EXTRA_ARGS+=(--mean)
  EXTRA_ARGS+=(--dummy-scans "$DUMMY_SCANS")
fi
if [[ "$REUSE_REFERENCE" == "1" ]]; then
  EXTRA_ARGS+=(--reuse-reference)
else
  EXTRA_ARGS+=(--no-reuse-reference)
fi

run_case "$OUT_SUBJ_REL" \
  --exp-root "$EXP_ROOT" \
  --out-root "$OUT_ROOT" \
  --out-subj "$OUT_SUBJ_REL" \
  --dwi-variant "$DWI_VARIANT" \
  --cut-token "$CUT_TOKEN" \
  "${EXTRA_ARGS[@]}" \
  --phantom-direct

finish_run
exit $?
