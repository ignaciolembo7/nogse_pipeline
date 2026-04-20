#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
BASE="Data-signals/20260122-PHANTOM_NISO4/QUALITY_JACK_19800122TMSF"
SRC="$BASE/QUALITY_JACK_19800122TMSF_001_NOGSE_CPMG_N2_TN50_G00_20260122092639_13"

FILES=(
  "niso4-1_mask.nii.gz"
  "niso4-2_mask.nii.gz"
  "niso4-3_mask.nii.gz"
  "niso4-4_mask.nii.gz"
  "niso4-5_mask.nii.gz"
)

if [[ ! -d "$BASE" ]]; then
  echo "ERROR: base directory not found: $BASE" >&2
  exit 1
fi

if [[ ! -d "$SRC" ]]; then
  echo "ERROR: source directory not found: $SRC" >&2
  exit 1
fi

for f in "${FILES[@]}"; do
  if [[ ! -f "$SRC/$f" ]]; then
    echo "ERROR: source file not found: $SRC/$f" >&2
    exit 1
  fi
done

echo "============================================================"
echo "Copy Selected Files"
echo "Base dir    : $BASE"
echo "Source dir  : $SRC"
echo "Files       : ${FILES[*]}"
echo "============================================================"

copied_dirs=0
copied_files=0

while read -r d; do
  [[ -z "$d" ]] && continue

  copied_dirs=$((copied_dirs + 1))
  echo "Target dir  : $d"

  for f in "${FILES[@]}"; do
    echo "  Copying   : $f"
    cp -av "$SRC/$f" "$d/"
    copied_files=$((copied_files + 1))
  done
done < <(find "$BASE" -mindepth 1 -maxdepth 1 -type d ! -path "$SRC" | sort)

echo
echo "Finished."
echo "  Target dirs : $copied_dirs"
echo "  Files copied: $copied_files"
