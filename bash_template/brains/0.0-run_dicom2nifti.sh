#!/usr/bin/env bash

set -u -o pipefail

# Driver script for the brain DICOM-to-NIfTI conversion batch.

SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_HOME/../.." && pwd)"

source "$SCRIPT_HOME/../helpers/dicom2nifti_batch_lib.sh"

PROJECT_ROOT="${PROJECT_ROOT:-$(resolve_default_project_root "$REPO_ROOT")}"
INPUT_ROOT="${INPUT_ROOT:-$PROJECT_ROOT/Data-DICOM}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/Data-NIFTI}"

init_run "run_brains_dicom2nifti"

run_case "20220622_BRAIN-1"
# run_case "20220624_BRAIN-2" "20220624_BRAIN2"
# run_case "202207081020_19760622MBBL"
# run_case "202207081630_19941220LUDG"
run_case "20230619_BRAIN-3"
run_case "20230623_BRAIN-4"
run_case "20230623_LUDG-2"
run_case "20230629_MBBL-2"
run_case "20230630_MBBL-3"
run_case "20230710_LUDG-3"

finish_run
exit $?
