#!/usr/bin/env bash

set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SUBJ=""
EXP_PARENT="Data-NIFTI-BRAINS-denoised_topup"
OUT_ROOT_REL="Data_signals"
CUT_TOKEN="_hifi_images_den"

usage() {
  echo "Usage:"
  echo "  bash nogse_pipeline/scripts/prep_phantom_b0.sh --subject <SUBJECT> [options]"
  echo
  echo "Options:"
  echo "  --subject SUBJ     Subject/experiment folder name"
  echo "                     Example: 20220622_BRAIN"
  echo "  --exp-parent DIR   Parent directory containing <SUBJECT>"
  echo "                     Default: Data-NIFTI-BRAINS-denoised_topup"
  echo "  --out-root DIR     Output root directory"
  echo "                     Default: Data_signals"
  echo "  --cut-token TOKEN  Token used to strip the NIfTI filename into seq_no_ext"
  echo "                     Default: _hifi_images_den"
  echo
  echo "Examples:"
  echo "  bash nogse_pipeline/scripts/prep_phantom_b0.sh \\"
  echo "    --subject 20220622_BRAIN"
  echo
  echo "  bash nogse_pipeline/scripts/prep_phantom_b0.sh \\"
  echo "    --subject 20220622_BRAIN \\"
  echo "    --exp-parent Data-NIFTI-BRAINS-denoised_topup"
  echo
  echo "  bash nogse_pipeline/scripts/prep_phantom_b0.sh \\"
  echo "    --subject 20220622_BRAIN \\"
  echo "    --exp-parent Data-NIFTI-PHANTOMS \\"
  echo "    --out-root Data_signals"
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
    -h|--help)
      usage
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      usage
      ;;
  esac
done

if [[ -z "$SUBJ" ]]; then
  echo "ERROR: --subject is required"
  usage
fi

PROJECT_ROOT="$PWD"
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
echo "============================================================"

shopt -s nullglob

found_any=0

for dwi in "$EXP_ROOT"/*.nii.gz; do
  found_any=1

  base="$(basename "$dwi")"
  seq_name_full="${base%.nii.gz}"
  seq_no_ext="${seq_name_full%%"$CUT_TOKEN"*}"

  bval="$EXP_ROOT/$seq_no_ext.bval"
  bvec="$EXP_ROOT/$seq_no_ext.bvec"

  if [[ ! -f "$bval" || ! -f "$bvec" ]]; then
    echo
    echo "[WARN] Could not find gradients for:"
    echo "       $seq_name_full"
    echo "       Expected:"
    echo "       $bval"
    echo "       $bvec"
    echo "       Skipping this sequence."
    continue
  fi

  out_seq="$OUT_SUBJ/$seq_no_ext"
  mkdir -p "$out_seq"

  nii_b0="$out_seq/NII_b0.nii.gz"
  b0_mean="$out_seq/b0_mean.nii.gz"

  echo
  echo "------------------------------------------------------------"
  echo "Sequence    : $seq_name_full"
  echo "SEQ_no_ext  : $seq_no_ext"
  echo "Output dir  : $out_seq"
  echo "------------------------------------------------------------"

  if [[ ! -f "$b0_mean" ]]; then
    echo "[INFO] Generating NII_b0 and b0_mean..."
    dwiextract -bzero "$dwi" "$nii_b0" --fslgrad "$bvec" "$bval" -force
    fslmaths "$nii_b0" -Tmean "$out_seq/b0_mean"
  else
    echo "[INFO] Already exists: $b0_mean"
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
echo "Now draw the masks manually on each:"
echo "  $OUT_ROOT_REL/$SUBJ/<sequence>/b0_mean.nii.gz"
echo "============================================================"
