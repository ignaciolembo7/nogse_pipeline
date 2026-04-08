#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
BASE="Data-signals/20220610-PHANTOM3"
SRC="$BASE/20220610-PHANTOM3_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d33_Hz000_b2000_DMRIPHANTOM_20220609151744_51"

FILES=(
  "fiber1_mask.nii.gz"
  "fiber2_mask.nii.gz"
  "water_mask.nii.gz"
  "water1_mask.nii.gz"
  "water2_mask.nii.gz"
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
