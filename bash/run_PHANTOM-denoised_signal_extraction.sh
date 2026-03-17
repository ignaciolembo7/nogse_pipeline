#!/usr/bin/env bash

set -u -o pipefail

SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_HOME/coreg_batch_lib.sh"

init_run "run_PHANTOM-denoised_signal_extraction"

run_case "" \
  --exp-root "$PROJECT_ROOT/Data-NIFTI-PHANTOM-denoised/20220613_PHANTOM4_PART1" \
  --out-root "$OUT_ROOT" \
  --cut-token "_den_grc" \
  --phantom-direct

finish_run
exit $?