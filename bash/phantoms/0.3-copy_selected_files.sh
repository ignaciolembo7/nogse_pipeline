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
  # "water1_mask.nii.gz"
  # "water2_mask.nii.gz"
  # "water3_mask.nii.gz"
)

find "$BASE" -mindepth 1 -maxdepth 1 -type d ! -path "$SRC" | while read -r d; do
  for f in "${FILES[@]}"; do
    cp -a "$SRC/$f" "$d/"
  done
done
