#!/usr/bin/env bash
set -euo pipefail

# Shared launcher for DICOM/Phoenix metadata extraction.

if [[ -z "${PROJECT_ROOT:-}" || -z "${REPO_ROOT:-}" ]]; then
  echo "ERROR: PROJECT_ROOT and REPO_ROOT must be defined by the caller." >&2
  exit 1
fi

PY="${PY:-python}"
SCRIPT_PATH="$REPO_ROOT/scripts/extract_dicom_sequence_metadata.py"

DICOM_ROOT="${DICOM_ROOT:-$PROJECT_ROOT/Data-DICOM}"
NIFTI_ROOT="${NIFTI_ROOT:-}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/analysis/dicom_metadata}"
DICOM_GLOB_CSV="${DICOM_GLOB_CSV:-*.IMA}"
NIFTI_GLOB_CSV="${NIFTI_GLOB_CSV:-*.nii.gz}"
RECURSIVE="${RECURSIVE:-0}"
NIFTI_RECURSIVE="${NIFTI_RECURSIVE:-0}"
WRITE_STRINGS="${WRITE_STRINGS:-0}"
SCANNER_GRAD_MAX_MTM="${SCANNER_GRAD_MAX_MTM:-80}"
OUT_XLSX="${OUT_XLSX:-$OUT_ROOT/sequence_metadata_from_dicom.xlsx}"
WRITE_PARQUET="${WRITE_PARQUET:-1}"

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "ERROR: extractor script not found: $SCRIPT_PATH" >&2
  exit 1
fi

if [[ ! -d "$DICOM_ROOT" ]]; then
  echo "ERROR: DICOM_ROOT does not exist: $DICOM_ROOT" >&2
  exit 1
fi

if [[ -n "$NIFTI_ROOT" && ! -d "$NIFTI_ROOT" ]]; then
  echo "ERROR: NIFTI_ROOT does not exist: $NIFTI_ROOT" >&2
  exit 1
fi

echo "============================================================"
echo "Extract DICOM sequence metadata"
echo "PROJECT_ROOT        : $PROJECT_ROOT"
echo "REPO_ROOT           : $REPO_ROOT"
echo "DICOM_ROOT          : $DICOM_ROOT"
echo "NIFTI_ROOT          : ${NIFTI_ROOT:-<not set>}"
echo "OUT_ROOT            : $OUT_ROOT"
echo "DICOM_GLOB_CSV      : $DICOM_GLOB_CSV"
echo "NIFTI_GLOB_CSV      : $NIFTI_GLOB_CSV"
echo "RECURSIVE           : $RECURSIVE"
echo "NIFTI_RECURSIVE     : $NIFTI_RECURSIVE"
echo "WRITE_STRINGS       : $WRITE_STRINGS"
echo "SCANNER_GRAD_MAX_MTM: $SCANNER_GRAD_MAX_MTM"
echo "WRITE_PARQUET       : $WRITE_PARQUET"
echo "OUT_XLSX            : $OUT_XLSX"
echo "PY                  : $PY"
echo "============================================================"

cmd=(
  "$PY" "$SCRIPT_PATH" "$DICOM_ROOT"
  --out-root "$OUT_ROOT"
  --scanner-grad-max-mtm "$SCANNER_GRAD_MAX_MTM"
  --out-xlsx "$OUT_XLSX"
)

IFS=',' read -r -a dicom_globs <<< "$DICOM_GLOB_CSV"
for pattern in "${dicom_globs[@]}"; do
  pattern="${pattern#"${pattern%%[![:space:]]*}"}"
  pattern="${pattern%"${pattern##*[![:space:]]}"}"
  if [[ -n "$pattern" ]]; then
    cmd+=(--glob "$pattern")
  fi
done

if [[ "$RECURSIVE" == "1" || "${RECURSIVE,,}" == "true" ]]; then
  cmd+=(--recursive)
fi

if [[ -n "$NIFTI_ROOT" ]]; then
  cmd+=(--nifti-root "$NIFTI_ROOT")

  IFS=',' read -r -a nifti_globs <<< "$NIFTI_GLOB_CSV"
  for pattern in "${nifti_globs[@]}"; do
    pattern="${pattern#"${pattern%%[![:space:]]*}"}"
    pattern="${pattern%"${pattern##*[![:space:]]}"}"
    if [[ -n "$pattern" ]]; then
      cmd+=(--nifti-glob "$pattern")
    fi
  done
fi

if [[ "$NIFTI_RECURSIVE" == "1" || "${NIFTI_RECURSIVE,,}" == "true" ]]; then
  cmd+=(--nifti-recursive)
fi

if [[ "$WRITE_STRINGS" == "1" || "${WRITE_STRINGS,,}" == "true" ]]; then
  cmd+=(--write-strings)
fi

if [[ "$WRITE_PARQUET" == "1" || "${WRITE_PARQUET,,}" == "true" ]]; then
  cmd+=(--write-parquet)
else
  cmd+=(--no-parquet)
fi

"${cmd[@]}"
