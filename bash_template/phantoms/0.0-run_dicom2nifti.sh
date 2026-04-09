#!/usr/bin/env bash

set -u -o pipefail

# Driver script for the phantom DICOM-to-NIfTI conversion batch.

SCRIPT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_HOME/../.." && pwd)"

source "$SCRIPT_HOME/../helpers/dicom2nifti_batch_lib.sh"

PROJECT_ROOT="${PROJECT_ROOT:-$(resolve_default_project_root "$REPO_ROOT")}"
INPUT_ROOT="${INPUT_ROOT:-$PROJECT_ROOT/Data-DICOM}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/Data-NIFTI}"

init_run "run_phantoms_dicom2nifti"

run_case "20220523_PHANTOMTEST"
run_case "20220607_GONZANY"
run_case "20220609-PHANTOM2"
run_case "20220610-PHANTOM3"
run_case "20220613_PHANTOM4_PART1"
run_case "20220613_PHANTOM4-PART2" "20220613_PHANTOM4_PART2"
run_case "20220616-PHANTOM5"
run_case "20220617-PHANTOM5-PART2"
run_case "20220622_PHANTOM6"

finish_run
exit $?
