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
<<<<<<< HEAD
EXP_ROOT="$PROJECT_ROOT/Data-NIFTI/20260122-PHANTOM_NISO4"
=======
EXP_ROOT="$PROJECT_ROOT/Data-NIFTI/20220610-PHANTOM3"
>>>>>>> origin/main
CUT_TOKEN=""
# Set to 1 only for acquisitions that must be collapsed into one mean image
# and one signal row per sequence.
USE_MEAN="1"

# Load shared batch helpers after defining all required variables.
source "$SCRIPT_HOME/../helpers/coreg_batch_lib.sh"

init_run "run_PHANTOM-denoised_signal_extraction"

EXTRA_ARGS=()
if [[ "$USE_MEAN" == "1" ]]; then
  EXTRA_ARGS+=(--mean)
fi

run_case "" \
  --exp-root "$EXP_ROOT" \
  --out-root "$OUT_ROOT" \
  --cut-token "$CUT_TOKEN" \
  "${EXTRA_ARGS[@]}" \
  --phantom-direct

finish_run
exit $?
