#!/usr/bin/env bash

set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SUBJ="20260122-PHANTOM_NISO4/QUALITY_JACK_19800122TMSF"
EXP_PARENT="Data-NIFTI"
DWI_GLOB="*_001_NOGSE*.nii.gz, *_002_NOGSE*.nii.gz"
DIR_X="1"
DIR_Y="0"
DIR_Z="0"
OVERWRITE=1
DRY_RUN=0
PY="${PY:-python}"

usage() {
  echo "Usage:"
  echo "  bash nogse_pipeline/bash_template/phantoms/0.1-run_make_gval_gvec.sh [options]"
  echo
  echo "Options:"
  echo "  --subject SUBJ      Subject/experiment folder name"
  echo "                      Default: $SUBJ"
  echo "  --exp-parent DIR    Parent directory containing <SUBJECT>"
  echo "                      Default: $EXP_PARENT"
  echo "  --glob PATTERN      Glob used to find the NIfTI files"
  echo "                      Default: $DWI_GLOB"
  echo "  --dir GX GY GZ      Common gradient direction written to each .gvec"
  echo "                      Default: $DIR_X $DIR_Y $DIR_Z"
  echo "  --overwrite         Overwrite existing .gval/.gvec files"
  echo "  --dry-run           Print what would be written without creating files"
  echo
  echo "Example:"
  echo "  bash nogse_pipeline/bash_template/phantoms/0.1-run_make_gval_gvec.sh"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject)
      SUBJ="$2"
      shift 2
      ;;
    --exp-parent)
      EXP_PARENT="$2"
      shift 2
      ;;
    --glob)
      DWI_GLOB="$2"
      shift 2
      ;;
    --dir)
      DIR_X="$2"
      DIR_Y="$3"
      DIR_Z="$4"
      shift 4
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      usage
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EXP_ROOT="$PROJECT_ROOT/$EXP_PARENT/$SUBJ"
SCRIPT_PATH="$PROJECT_ROOT/nogse_pipeline/scripts/make_gval_gvec_from_filenames.py"

if [[ ! -d "$EXP_ROOT" ]]; then
  echo "ERROR: experiment root does not exist:"
  echo "  $EXP_ROOT"
  exit 1
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "ERROR: generator script does not exist:"
  echo "  $SCRIPT_PATH"
  exit 1
fi

echo "============================================================"
echo "Generate gval/gvec"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "EXP_ROOT    : $EXP_ROOT"
echo "DWI_GLOB    : $DWI_GLOB"
echo "DIRECTION   : $DIR_X $DIR_Y $DIR_Z"
echo "OVERWRITE   : $OVERWRITE"
echo "DRY_RUN     : $DRY_RUN"
echo "PY          : $PY"
echo "============================================================"

CMD=(
  "$PY" "$SCRIPT_PATH" "$EXP_ROOT"
  --glob "$DWI_GLOB"
  --direction "$DIR_X" "$DIR_Y" "$DIR_Z"
)

if [[ "$OVERWRITE" -eq 1 ]]; then
  CMD+=(--overwrite)
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  CMD+=(--dry-run)
fi

"${CMD[@]}"
