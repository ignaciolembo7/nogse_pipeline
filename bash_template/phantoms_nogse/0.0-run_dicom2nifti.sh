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

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/phantoms-3_nogse}"

run_case "20260505_PHANTOM_FIBER"
# ------------------------------------------------------------------
# ------------------------------------------------------------------

finish_run
exit $?
