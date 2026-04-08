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

# Load shared batch helpers after defining all required variables.
source "$SCRIPT_HOME/../coreg_batch_lib.sh"

init_run "run_PHANTOM-denoised_signal_extraction"

run_case "" \
  --exp-root "$PROJECT_ROOT/Data-NIFTI-PHANTOM-denoised/20220610-PHANTOM3" \
  --out-root "$OUT_ROOT" \
  --cut-token "_den_grc" \
  --phantom-direct

finish_run
exit $?
