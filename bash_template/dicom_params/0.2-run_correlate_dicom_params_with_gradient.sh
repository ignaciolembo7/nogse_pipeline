#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
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
NAME="${NAME:-QUALITY_JACK_19800122TMSF}"
METADATA_ROOT="${METADATA_ROOT:-$PROJECT_ROOT/analysis/dicom_metadata/$EXPERIMENT/$NAME}"
KEY_VALUES="${KEY_VALUES:-$METADATA_ROOT/dicom_asconv_key_values.long.parquet}"
NIFTI_TABLE="${NIFTI_TABLE:-$METADATA_ROOT/sequence_parameters_by_nifti_from_dicom.parquet}"
OUT_CSV="${OUT_CSV:-}"
OUT_XLSX="${OUT_XLSX:-}"
MIN_OBSERVATIONS="${MIN_OBSERVATIONS:-3}"
SORT_BY="${SORT_BY:-abs_correlation}"
# ------------------------------------------------------------------
# ------------------------------------------------------------------

SCRIPT_PATH="$REPO_ROOT/scripts/dicom_correlate_asconv_with_gradient.py"

if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "ERROR: script not found: $SCRIPT_PATH" >&2
    exit 1
fi
if [[ ! -f "$KEY_VALUES" ]]; then
    echo "ERROR: KEY_VALUES does not exist: $KEY_VALUES" >&2
    echo "Run 0.0-run_extract_dicom_sequence_metadata.sh first to extract CSV and Parquet outputs." >&2
    exit 1
fi
if [[ ! -f "$NIFTI_TABLE" ]]; then
    echo "ERROR: NIFTI_TABLE does not exist: $NIFTI_TABLE" >&2
    exit 1
fi

echo "============================================================"
echo "Correlate DICOM ASCCONV parameters with gradient"
echo "KEY_VALUES      : $KEY_VALUES"
echo "NIFTI_TABLE     : $NIFTI_TABLE"
echo "MIN_OBSERVATIONS: $MIN_OBSERVATIONS"
echo "SORT_BY         : $SORT_BY"
echo "OUT_CSV         : ${OUT_CSV:-<auto>}"
echo "OUT_XLSX        : ${OUT_XLSX:-<auto>}"
echo "PY              : $PY"
echo "============================================================"

cmd=(
    "$PY" "$SCRIPT_PATH" "$KEY_VALUES"
    --nifti-table "$NIFTI_TABLE"
    --min-observations "$MIN_OBSERVATIONS"
    --sort-by "$SORT_BY"
)

if [[ -n "$OUT_CSV" ]]; then
    cmd+=(--out-csv "$OUT_CSV")
fi
if [[ -n "$OUT_XLSX" ]]; then
    cmd+=(--out-xlsx "$OUT_XLSX")
fi

"${cmd[@]}"
