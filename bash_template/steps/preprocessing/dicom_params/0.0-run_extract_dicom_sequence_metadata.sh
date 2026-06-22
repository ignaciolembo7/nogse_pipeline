#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
REPO_ROOT="$PROJECT_ROOT/nogse_pipeline"

DEFAULT_PY="python"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    DEFAULT_PY="${CONDA_PREFIX}/bin/python"
elif [[ -x "/home/ignacio.lemboferrari@unitn.it/.conda/envs/nogse_pipe_env/bin/python" ]]; then
    DEFAULT_PY="/home/ignacio.lemboferrari@unitn.it/.conda/envs/nogse_pipe_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    DEFAULT_PY="$(command -v python3)"
fi

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
PY="${PY:-$DEFAULT_PY}"
EXPERIMENT="${EXPERIMENT:-20260506_PHANTOM_FIBER}"
NIFTI_EXPERIMENT="${NIFTI_EXPERIMENT:-20260506-PHANTOM_FIBER}"
NAME="${NAME:-QUALITY_JACK_19800122TMSF}"
DICOM_ROOT="${DICOM_ROOT:-$PROJECT_ROOT/Data-DICOM/$EXPERIMENT/$NAME}"
NIFTI_ROOT="${NIFTI_ROOT:-$PROJECT_ROOT/Data-NIFTI/$NIFTI_EXPERIMENT/$NAME}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/analysis/dicom_metadata/$EXPERIMENT/$NAME}"
DICOM_GLOB_CSV="${DICOM_GLOB_CSV:-*.IMA}"
NIFTI_GLOB_CSV="${NIFTI_GLOB_CSV:-*_NOGSE*.nii.gz}"
RECURSIVE="${RECURSIVE:-0}"
NIFTI_RECURSIVE="${NIFTI_RECURSIVE:-0}"
WRITE_STRINGS="${WRITE_STRINGS:-0}"
SCANNER_GRAD_MAX_MTM="${SCANNER_GRAD_MAX_MTM:-80}"
WRITE_PARQUET="${WRITE_PARQUET:-1}"
# ------------------------------------------------------------------
# ------------------------------------------------------------------

source "$REPO_ROOT/bash_template/helpers/run_extract_dicom_sequence_metadata.sh"
