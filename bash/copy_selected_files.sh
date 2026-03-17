#!/usr/bin/env bash

BASE="Data-signals/20220613_PHANTOM4_PART1"
SRC="$BASE/20220613_PHANTOM4_PART1_ep2d_advdiff_AP_919D_OGSE_10bval_3orthodir_d40_Hz000_b2000_DMRIPHANTOM_20220613100550_81"

FILES=(
  "fiber1_mask.nii.gz"
  "fiber2_mask.nii.gz"
  "water1_mask.nii.gz"
  "water2_mask.nii.gz"
  "water3_mask.nii.gz"
)

find "$BASE" -mindepth 1 -maxdepth 1 -type d ! -path "$SRC" | while read -r d; do
  for f in "${FILES[@]}"; do
    cp -a "$SRC/$f" "$d/"
  done
done