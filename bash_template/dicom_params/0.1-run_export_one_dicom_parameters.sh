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

# Edit only this value for the usual workflow. Full path, basename, stem,
# or a unique substring are accepted.
DICOM_FILE="${DICOM_FILE:-QUALITY_JACK.MR.JOVICICHJORGE_IGNACIOLEMBOFERRARI_NOGSE.0060.0001.2026.05.06.17.21.33.307329.523834585.IMA}"

KEY_VALUES_PARQUET="${KEY_VALUES_PARQUET:-$METADATA_ROOT/dicom_asconv_key_values.long.parquet}"
KEY_VALUES_CSV="${KEY_VALUES_CSV:-$METADATA_ROOT/dicom_asconv_key_values.long.csv}"
OUT_CSV="${OUT_CSV:-}"
OUT_PARQUET="${OUT_PARQUET:-}"
OUT_XLSX="${OUT_XLSX:-}"
# ------------------------------------------------------------------
# ------------------------------------------------------------------

EXPORT_SCRIPT="$REPO_ROOT/scripts/dicom_export_file_parameters.py"

if [[ ! -f "$EXPORT_SCRIPT" ]]; then
    echo "ERROR: export script not found: $EXPORT_SCRIPT" >&2
    exit 1
fi
if [[ ! -d "$METADATA_ROOT" ]]; then
    echo "ERROR: METADATA_ROOT does not exist: $METADATA_ROOT" >&2
    echo "Run DICOM metadata extraction first." >&2
    exit 1
fi

KEY_VALUES="$KEY_VALUES_PARQUET"
if [[ ! -f "$KEY_VALUES" ]]; then
    if [[ -f "$KEY_VALUES_CSV" ]]; then
        KEY_VALUES="$KEY_VALUES_CSV"
    else
        echo "ERROR: neither key-value Parquet nor CSV exists:" >&2
        echo "  $KEY_VALUES_PARQUET" >&2
        echo "  $KEY_VALUES_CSV" >&2
        echo "Run 0.0-run_extract_dicom_sequence_metadata.sh first." >&2
        exit 1
    fi
fi

echo "============================================================"
echo "Export one DICOM ASCCONV parameter table"
echo "METADATA_ROOT     : $METADATA_ROOT"
echo "KEY_VALUES        : $KEY_VALUES"
echo "DICOM_FILE        : $DICOM_FILE"
echo "OUT_CSV           : ${OUT_CSV:-<disabled>}"
echo "OUT_PARQUET       : ${OUT_PARQUET:-<auto>}"
echo "OUT_XLSX          : ${OUT_XLSX:-<auto>}"
echo "PY                : $PY"
echo "============================================================"

cmd=(
    "$PY" "$EXPORT_SCRIPT" "$KEY_VALUES"
    --dicom-file "$DICOM_FILE"
)

if [[ -n "$OUT_CSV" ]]; then
    cmd+=(--out-csv "$OUT_CSV")
fi
if [[ -n "$OUT_PARQUET" ]]; then
    cmd+=(--out-parquet "$OUT_PARQUET")
fi
if [[ -n "$OUT_XLSX" ]]; then
    cmd+=(--out-xlsx "$OUT_XLSX")
fi

"${cmd[@]}"
