#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/helpers/master_table_common.sh"
pipeline_maybe_step_help contrast_resampled "$@"
pipeline_setup_common
pipeline_set_dataset_defaults "${TYPE_SUBJ:-${DATASET:?TYPE_SUBJ or DATASET is required}}"

EXPORT_RESAMPLED_SCRIPT="${EXPORT_RESAMPLED_SCRIPT:-$REPO_ROOT/scripts/fitting/export_ogse_resampled_contrasts_from_fits.py}"

# Signal fits root from step 04 (fit_signals or fit_signals_global) — required.
if [[ -z "${SIGNAL_FIT_OUT_ROOT:-}" ]]; then
    echo "ERROR: SIGNAL_FIT_OUT_ROOT must be set to the signal fits directory from step 04." >&2
    echo "  Run step 04 (fit_signals or fit_signals_global) first, then set:" >&2
    echo "  SIGNAL_FIT_OUT_ROOT=<path to fits directory>" >&2
    exit 1
fi
if [[ ! -d "$SIGNAL_FIT_OUT_ROOT" ]]; then
    echo "ERROR: signal fits directory not found: $SIGNAL_FIT_OUT_ROOT" >&2
    echo "  Run step 04 (fit_signals or fit_signals_global) first." >&2
    exit 1
fi

CONTRAST_ROOT="${CONTRAST_ROOT:-$ANALYSIS_ROOT/contrast-data-master}"
CONTRAST_RESAMPLED_OUT_DIR="${CONTRAST_RESAMPLED_OUT_DIR:-$ANALYSIS_ROOT/contrast-data-resampled}"
CONTRAST_RESAMPLED_PARQUET="${CONTRAST_RESAMPLED_PARQUET:-$CONTRAST_RESAMPLED_OUT_DIR/master_contrast_resampled.parquet}"

pipeline_require_file "$EXPORT_RESAMPLED_SCRIPT" "export resampled contrasts script"
mkdir -p "$CONTRAST_RESAMPLED_OUT_DIR"

"$PY" "$EXPORT_RESAMPLED_SCRIPT" "$SIGNAL_FIT_OUT_ROOT" \
    --contrast-root "$CONTRAST_ROOT" \
    --out-dir "$CONTRAST_RESAMPLED_OUT_DIR" \
    --grid-min-mTm "${EXPORT_GRID_MIN_MTM:-0}" \
    --grid-max-mode "${EXPORT_GRID_MAX_MODE:-observed_pair_max}" \
    --grid-n "${EXPORT_GRID_N:-100}" \
    --contrast-mode "${EXPORT_CONTRAST_MODE:-signal_fit}" \
    ${EXPORT_MODELS:+--models $EXPORT_MODELS} \
    ${EXPORT_G_MIN_DERIVED:+--g-min-derived "$EXPORT_G_MIN_DERIVED"} \
    ${EXPORT_RESAMPLED_EXTRA_ARGS:-}

echo "Concatenating resampled contrasts → $CONTRAST_RESAMPLED_PARQUET"
"$PY" - "$CONTRAST_RESAMPLED_OUT_DIR" "$CONTRAST_RESAMPLED_PARQUET" <<'PY'
import sys, pandas as pd
from pathlib import Path
files = sorted(Path(sys.argv[1]).rglob("resampled_contrasts.long.parquet"))
if not files:
    print("WARNING: no resampled contrast parquets found to concatenate.", file=sys.stderr)
    sys.exit(1)
pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).to_parquet(sys.argv[2], index=False)
PY
echo "Done. master_contrast_resampled: $CONTRAST_RESAMPLED_PARQUET"
