#!/usr/bin/env bash

set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SUBJ="20260122-PHANTOM_FIBER/QUALITY_JACK_19800122TMSF"
EXP_PARENT="Data-NIFTI"
OUT_ROOT_REL="Data-signals"
CUT_TOKEN=""
# Which dcm2niix conflict variant to use: "none", "a", "b", ... or "all".
DWI_VARIANT="none"
# Reference mode:
#   - mean: average all DWI volumes and write NII_mean.nii.gz + mean.nii.gz
#   - b0  : extract only b=0 volumes and write NII_b0.nii.gz + b0_mean.nii.gz
REF_MODE="mean"
# Number of initial volumes to discard when REF_MODE="mean".
DUMMY_SCANS="5"
# Set to 1 to reuse existing reference images, or 0 to overwrite them.
REUSE_REFERENCE="0"

usage() {
  echo "Usage:"
  echo "  bash nogse_pipeline/bash_template/phantoms/0.2-prep_phantom_b0.sh [options]"
  echo
  echo "Options:"
  echo "  --subject SUBJ     Subject/experiment folder name"
  echo "                     Default: $SUBJ"
  echo "  --exp-parent DIR   Parent directory containing <SUBJECT>"
  echo "                     Default: $EXP_PARENT"
  echo "  --out-root DIR     Output root directory"
  echo "                     Default: $OUT_ROOT_REL"
  echo "  --cut-token TOKEN  Token used to strip the NIfTI filename into seq_no_ext"
  echo "                     Default: '$CUT_TOKEN'"
  echo "  --dwi-variant VAR  NIfTI variant to use: none, a, b, ... or all"
  echo "                     Default: $DWI_VARIANT"
  echo
  echo "Reference mode is configured inside the script:"
  echo "  REF_MODE=\"mean\"   -> average all volumes"
  echo "  REF_MODE=\"b0\"     -> average only b=0 volumes"
  echo "  DUMMY_SCANS=\"0\"   -> discard no initial volumes for REF_MODE=\"mean\""
  echo "  REUSE_REFERENCE=\"1\" -> reuse existing reference images"
  echo
  echo "Examples:"
  echo "  bash nogse_pipeline/bash_template/phantoms/0.2-prep_phantom_b0.sh"
  echo
  echo "  bash nogse_pipeline/bash_template/phantoms/0.2-prep_phantom_b0.sh \\"
  echo "    --subject 20260122-PHANTOM_NISO4/QUALITY_JACK_19800122TMSF"
  echo
  echo "  bash nogse_pipeline/bash_template/phantoms/0.2-prep_phantom_b0.sh \\"
  echo "    --subject 20260122-PHANTOM_NISO4/QUALITY_JACK_19800122TMSF \\"
  echo "    --exp-parent Data-NIFTI \\"
  echo "    --out-root Data-signals"
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
    --out-root)
      OUT_ROOT_REL="$2"
      shift 2
      ;;
    --cut-token)
      CUT_TOKEN="$2"
      shift 2
      ;;
    --dwi-variant)
      DWI_VARIANT="$2"
      shift 2
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

case "$REF_MODE" in
  mean|b0)
    ;;
  *)
    echo "ERROR: REF_MODE must be 'mean' or 'b0', got: $REF_MODE"
    exit 1
    ;;
esac

if ! [[ "$DUMMY_SCANS" =~ ^[0-9]+$ ]]; then
  echo "ERROR: DUMMY_SCANS must be a non-negative integer, got: $DUMMY_SCANS"
  exit 1
fi

case "$REUSE_REFERENCE" in
  0|1)
    ;;
  *)
    echo "ERROR: REUSE_REFERENCE must be 0 or 1, got: $REUSE_REFERENCE"
    exit 1
    ;;
esac

case "${DWI_VARIANT,,}" in
  all|any|"*"|none|primary|base|[a-z])
    ;;
  *)
    echo "ERROR: DWI_VARIANT must be 'none', 'a', 'b', ..., or 'all', got: $DWI_VARIANT"
    exit 1
    ;;
esac

dcm2niix_variant() {
  local seq_name="$1"
  if [[ "$seq_name" =~ _[0-9]+([A-Za-z])$ ]]; then
    printf '%s\n' "${BASH_REMATCH[1],,}"
  else
    printf '\n'
  fi
}

dwi_variant_matches() {
  local seq_name="$1"
  local wanted="${2,,}"
  local variant
  if [[ "$wanted" == "all" || "$wanted" == "any" || "$wanted" == "*" ]]; then
    return 0
  fi
  variant="$(dcm2niix_variant "$seq_name")"
  if [[ "$wanted" == "none" || "$wanted" == "primary" || "$wanted" == "base" || -z "$wanted" ]]; then
    [[ -z "$variant" ]]
    return
  fi
  [[ "$variant" == "$wanted" ]]
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EXP_ROOT="$PROJECT_ROOT/$EXP_PARENT/$SUBJ"
OUT_SUBJ="$PROJECT_ROOT/$OUT_ROOT_REL/$SUBJ"

if [[ ! -d "$EXP_ROOT" ]]; then
  echo "ERROR: experiment root does not exist:"
  echo "  $EXP_ROOT"
  exit 1
fi

mkdir -p "$OUT_SUBJ"

echo "============================================================"
echo "SUBJECT     : $SUBJ"
echo "PROJECT_ROOT: $PROJECT_ROOT"
echo "EXP_PARENT  : $EXP_PARENT"
echo "EXP_ROOT    : $EXP_ROOT"
echo "OUT_SUBJ    : $OUT_SUBJ"
echo "CUT_TOKEN   : $CUT_TOKEN"
echo "DWI_VARIANT : $DWI_VARIANT"
echo "REF_MODE    : $REF_MODE"
echo "DUMMY_SCANS : $DUMMY_SCANS"
echo "REUSE_REF   : $REUSE_REFERENCE"
echo "============================================================"

shopt -s nullglob

found_any=0

for dwi in "$EXP_ROOT"/*.nii.gz; do
  base="$(basename "$dwi")"
  seq_name_full="${base%.nii.gz}"
  if ! dwi_variant_matches "$seq_name_full" "$DWI_VARIANT"; then
    echo "[SKIP] Variant excluded by DWI_VARIANT=$DWI_VARIANT: $seq_name_full"
    continue
  fi
  found_any=1

  if [[ -n "$CUT_TOKEN" ]]; then
    seq_no_ext="${seq_name_full%%"$CUT_TOKEN"*}"
  else
    seq_no_ext="$seq_name_full"
  fi

  out_seq="$OUT_SUBJ/$seq_no_ext"
  mkdir -p "$out_seq"

  echo
  echo "------------------------------------------------------------"
  echo "Sequence    : $seq_name_full"
  echo "SEQ_no_ext  : $seq_no_ext"
  echo "Output dir  : $out_seq"
  echo "------------------------------------------------------------"

  if [[ "$REF_MODE" == "mean" ]]; then
    nii_mean="$out_seq/NII_mean.nii.gz"
    mean_img="$out_seq/mean.nii.gz"

    if [[ "$REUSE_REFERENCE" == "0" || ! -f "$mean_img" ]]; then
      echo "[INFO] Generating NII_mean and mean..."
      if [[ "$DUMMY_SCANS" -gt 0 ]]; then
        nvol="$(fslnvols "$dwi")"
        if [[ "$DUMMY_SCANS" -ge "$nvol" ]]; then
          echo "ERROR: DUMMY_SCANS must be smaller than the number of DWI volumes."
          echo "       DUMMY_SCANS=$DUMMY_SCANS, nvol=$nvol"
          exit 1
        fi
        fslroi "$dwi" "$nii_mean" "$DUMMY_SCANS" -1
      else
        cp -f "$dwi" "$nii_mean"
      fi
      fslmaths "$nii_mean" -Tmean "$out_seq/mean"
    else
      echo "[INFO] Already exists: $mean_img"
    fi
  else
    bval="$EXP_ROOT/$seq_no_ext.bval"
    bvec="$EXP_ROOT/$seq_no_ext.bvec"
    nii_b0="$out_seq/NII_b0.nii.gz"
    b0_mean="$out_seq/b0_mean.nii.gz"

    if [[ ! -f "$bval" || ! -f "$bvec" ]]; then
      echo
      echo "[WARN] Could not find gradients for:"
      echo "       $seq_name_full"
      echo "       Expected:"
      echo "       $bval"
      echo "       $bvec"
      echo "       Set REF_MODE=\"mean\" inside this script to average all volumes instead."
      echo "       Skipping this sequence."
      continue
    fi

    if [[ "$REUSE_REFERENCE" == "0" || ! -f "$b0_mean" ]]; then
      echo "[INFO] Generating NII_b0 and b0_mean..."
      dwiextract -bzero "$dwi" "$nii_b0" --fslgrad "$bvec" "$bval" -force
      fslmaths "$nii_b0" -Tmean "$out_seq/b0_mean"
    else
      echo "[INFO] Already exists: $b0_mean"
    fi
  fi
done

if [[ "$found_any" -eq 0 ]]; then
  echo "ERROR: no .nii.gz files were found in:"
  echo "  $EXP_ROOT"
  exit 1
fi

echo
echo "============================================================"
echo "Done."
if [[ "$REF_MODE" == "mean" ]]; then
  echo "Now draw the masks manually on each:"
  echo "  $OUT_ROOT_REL/$SUBJ/<sequence>/mean.nii.gz"
else
  echo "Now draw the masks manually on each:"
  echo "  $OUT_ROOT_REL/$SUBJ/<sequence>/b0_mean.nii.gz"
fi
echo "============================================================"
