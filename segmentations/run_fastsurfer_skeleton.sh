#!/usr/bin/env bash

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Portable whole-corpus-callosum skeletonization runner for FastSurfer-CC outputs.
#
# This script does NOT use MRtrix for skeletonization, because the public
# mrtrix3/mrtrix3 image exposes maskfilter filters such as clean/connect/dilate/
# erode/median, but not thin/skeletonization.
#
# The script:
#   1) Selects labels 251..255 together from callosum.CC.orig.mgz.
#   2) Converts the whole corpus callosum binary mask to NIfTI.
#   3) Skeletonizes the whole corpus callosum mask in 3D using scikit-image.
#   4) Optionally dilates the skeleton by N voxels.
#   5) Clips the skeleton or thickened skeleton back to the original CC mask.
#   6) Converts the binary skeleton mask back to MGZ.
#   7) Applies that skeleton mask to the original multi-label segmentation.
#   8) Preserves the original internal labels 251..255 on the skeleton.
#
# Usage:
#   bash run_fastsurfer_skeleton.sh SUBJECT_ID [SKELETON_DILATE_STEPS]
#
# Examples:
#   bash run_fastsurfer_skeleton.sh sub-20220622_BRAIN 0
#   bash run_fastsurfer_skeleton.sh sub-20220622_BRAIN 1
#
# Main outputs with the default INPUT_NAME=callosum.CC.orig.mgz:
#   SKELETON_DILATE_STEPS=0 -> callosum.CC.orig.sket.mgz
#   SKELETON_DILATE_STEPS=1 -> callosum.CC.orig.sket1.mgz
#   SKELETON_DILATE_STEPS=2 -> callosum.CC.orig.sket2.mgz
#
# Optional environment variables:
#   IMAGE          FreeSurfer/FastSurfer Docker image.
#   PYTHON_IMAGE   Local Python image used for skeletonization.
#   PROJECT_DIR    Root FastSurfer data directory.
#   OUTPUT_DIR     FastSurfer subjects directory. Defaults to PROJECT_DIR/OUT.
#   INPUT_NAME     Input segmentation file in the subject mri directory.
#                  Defaults to callosum.CC.orig.mgz.
#   OUTPUT_NAME    Output skeleton file in the subject mri directory.
#                  By default, this is derived from INPUT_NAME and
#                  SKELETON_DILATE_STEPS.
#   SKELETON_DILATE_STEPS
#                  Optional skeleton thickening radius in voxels.
#                  0 gives a pure skeleton. 1 gives a slightly thicker
#                  medial band. Defaults to 0.
#   FORCE          Set to 1 to overwrite existing skeleton outputs.
#   REBUILD_PYTHON_IMAGE
#                  Set to 1 to rebuild the local Python image.
# ---------------------------------------------------------------------------

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '3,60p' "${BASH_SOURCE[0]}"
    exit 0
fi

SUBJECT_ID="${1:-${SUBJECT_ID:-BRAIN-1}}"
SKELETON_DILATE_STEPS="${2:-${SKELETON_DILATE_STEPS:-0}}"
IMAGE="${IMAGE:-deepmi/fastsurfer:cpu-v2.5.4}"
PYTHON_IMAGE="${PYTHON_IMAGE:-cc-skeleton-python:latest}"
INPUT_NAME="${INPUT_NAME:-callosum.CC.orig.mgz}"
OUTPUT_NAME_WAS_SET=0
if [[ -n "${OUTPUT_NAME+x}" ]]; then
    OUTPUT_NAME_WAS_SET=1
fi
FORCE="${FORCE:-0}"
REBUILD_PYTHON_IMAGE="${REBUILD_PYTHON_IMAGE:-0}"

if [[ ! "${SUBJECT_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Error: SUBJECT_ID may contain only letters, numbers, dots, underscores, and hyphens."
    exit 1
fi

if [[ ! "${SKELETON_DILATE_STEPS}" =~ ^[0-9]+$ ]]; then
    echo "Error: SKELETON_DILATE_STEPS must be a non-negative integer."
    exit 1
fi

if [[ ! "${INPUT_NAME}" =~ ^[A-Za-z0-9._-]+\.mgz$ ]]; then
    echo "Error: INPUT_NAME must be a simple .mgz filename."
    exit 1
fi

INPUT_STEM="${INPUT_NAME%.mgz}"

if [[ "${OUTPUT_NAME_WAS_SET}" -eq 0 ]]; then
    if [[ "${SKELETON_DILATE_STEPS}" -eq 0 ]]; then
        OUTPUT_NAME="${INPUT_STEM}.sket.mgz"
    else
        OUTPUT_NAME="${INPUT_STEM}.sket${SKELETON_DILATE_STEPS}.mgz"
    fi
fi

if [[ ! "${OUTPUT_NAME}" =~ ^[A-Za-z0-9._-]+\.mgz$ ]]; then
    echo "Error: OUTPUT_NAME must be a simple .mgz filename."
    exit 1
fi

if [[ "${FORCE}" != "0" && "${FORCE}" != "1" ]]; then
    echo "Error: FORCE must be 0 or 1."
    exit 1
fi

if [[ "${REBUILD_PYTHON_IMAGE}" != "0" && "${REBUILD_PYTHON_IMAGE}" != "1" ]]; then
    echo "Error: REBUILD_PYTHON_IMAGE must be 0 or 1."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}/FASTSURFER_CC}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/OUT}"
LICENSE_FILE="${PROJECT_DIR}/license.txt"

SUBJECT_DIR="${OUTPUT_DIR}/${SUBJECT_ID}"
MRI_DIR="${SUBJECT_DIR}/mri"
INPUT_LABELS="${MRI_DIR}/${INPUT_NAME}"
FINAL_LABELS="${MRI_DIR}/${OUTPUT_NAME}"
SKELETON_DIR="${MRI_DIR}/cc_skeleton"

OUTPUT_STEM="${OUTPUT_NAME%.mgz}"

WHOLE_CC_MASK_MGZ="${SKELETON_DIR}/${INPUT_STEM}.whole.mask.mgz"
WHOLE_CC_MASK_NII="${SKELETON_DIR}/${INPUT_STEM}.whole.mask.nii.gz"
SKELETON_BIN_NII="${SKELETON_DIR}/${OUTPUT_STEM}.bin.nii.gz"
SKELETON_BIN_MGZ="${SKELETON_DIR}/${OUTPUT_STEM}.bin.mgz"
STATS_BEFORE="${SKELETON_DIR}/${INPUT_STEM}.stats.before.txt"
STATS_AFTER="${SKELETON_DIR}/${OUTPUT_STEM}.stats.after.txt"

LOG_DIR="${PROJECT_DIR}/LOGS/CC_SKELETON"
PROVENANCE_ROOT="${PROJECT_DIR}/PROVENANCE/CC_SKELETON"
RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="${LOG_DIR}/${SUBJECT_ID}_${OUTPUT_STEM}_skeleton_${RUN_TIMESTAMP}.log"
PROVENANCE_DIR="${PROVENANCE_ROOT}/${SUBJECT_ID}/${INPUT_STEM}/skeleton_dilate${SKELETON_DILATE_STEPS}/${RUN_TIMESTAMP}"

mkdir -p "${SKELETON_DIR}" "${LOG_DIR}" "${PROVENANCE_DIR}"

# Save all output to a timestamped log while keeping it visible in the terminal.
exec > >(tee -a "${LOG_FILE}") 2>&1

on_exit() {
    local status=$?
    echo
    if [[ ${status} -eq 0 ]]; then
        echo "Run completed successfully: $(date --iso-8601=seconds 2>/dev/null || date)"
    else
        echo "Run failed with exit code ${status}: $(date --iso-8601=seconds 2>/dev/null || date)"
    fi
    echo "Log file: ${LOG_FILE}"
}
trap on_exit EXIT

UNAME_S="$(uname -s 2>/dev/null || echo unknown)"

case "${UNAME_S}" in
    MINGW*|MSYS*|CYGWIN*)
        PLATFORM="git-bash"
        ;;
    Darwin*)
        PLATFORM="macos"
        ;;
    Linux*)
        if grep -qi microsoft /proc/version 2>/dev/null; then
            PLATFORM="wsl"
        else
            PLATFORM="linux"
        fi
        ;;
    *)
        PLATFORM="unknown"
        ;;
esac

# Git Bash rewrites arguments beginning with "/" before passing them to
# Windows executables. Disable that behavior for Linux container paths.
docker_safe() {
    if [[ "${PLATFORM}" == "git-bash" ]]; then
        MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" docker "$@"
    else
        docker "$@"
    fi
}

# Bind-mount source paths must use the host operating system path format.
to_docker_host_path() {
    local path="$1"

    if [[ "${PLATFORM}" == "git-bash" ]]; then
        if ! command -v cygpath >/dev/null 2>&1; then
            echo "Error: cygpath is required when running from Git Bash."
            return 1
        fi
        cygpath -m "${path}"
    else
        printf '%s\n' "${path}"
    fi
}

if ! command -v docker >/dev/null 2>&1; then
    echo "Error: Docker is not available in this terminal."
    exit 1
fi

if ! docker_safe info >/dev/null 2>&1; then
    echo "Error: Docker Desktop or the Docker daemon is not running."
    exit 1
fi

if [[ ! -f "${LICENSE_FILE}" ]]; then
    echo "Error: FreeSurfer license not found:"
    echo "  ${LICENSE_FILE}"
    exit 1
fi

if [[ ! -f "${INPUT_LABELS}" ]]; then
    echo "Error: FastSurfer-CC segmentation not found:"
    echo "  ${INPUT_LABELS}"
    exit 1
fi

if [[ -f "${FINAL_LABELS}" && "${FORCE}" != "1" ]]; then
    echo "Error: skeletonized corpus callosum segmentation already exists:"
    echo "  ${FINAL_LABELS}"
    echo
    echo "To overwrite it, run:"
    echo "  FORCE=1 bash $(basename "${BASH_SOURCE[0]}") ${SUBJECT_ID} ${SKELETON_DILATE_STEPS}"
    exit 1
fi

DOCKER_MRI_DIR="$(to_docker_host_path "${MRI_DIR}")"
DOCKER_LICENSE_FILE="$(to_docker_host_path "${LICENSE_FILE}")"

DOCKER_USER_ARGS=()

if [[ "${PLATFORM}" == "git-bash" ]]; then
    # Windows bind mounts do not benefit from Linux UID/GID mapping.
    DOCKER_USER_ARGS=(--user root)
elif [[ "$(id -u)" -eq 0 ]]; then
    DOCKER_USER_ARGS=(--user root)
else
    # Avoid root-owned output files on Linux, macOS, and WSL.
    DOCKER_USER_ARGS=(--user "$(id -u):$(id -g)")
fi

build_python_image_if_needed() {
    if [[ "${REBUILD_PYTHON_IMAGE}" == "0" ]] && docker_safe image inspect "${PYTHON_IMAGE}" >/dev/null 2>&1; then
        echo "Python skeletonization image already available: ${PYTHON_IMAGE}"
        return 0
    fi

    echo "Building local Python skeletonization image: ${PYTHON_IMAGE}"
    echo "This is only needed the first time, or when REBUILD_PYTHON_IMAGE=1."

    docker_safe build -t "${PYTHON_IMAGE}" - <<'DOCKERFILE'
FROM python:3.11-slim
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir nibabel numpy scikit-image
DOCKERFILE
}

echo "Run started: $(date --iso-8601=seconds 2>/dev/null || date)"
echo "Platform: ${PLATFORM}"
echo "Subject ID: ${SUBJECT_ID}"
echo "Input labels: ${INPUT_LABELS}"
echo "Skeleton dilation steps: ${SKELETON_DILATE_STEPS}"
echo "Final output: ${FINAL_LABELS}"
echo "FreeSurfer/FastSurfer image: ${IMAGE}"
echo "Python skeletonization image: ${PYTHON_IMAGE}"
echo "MRI mount: ${DOCKER_MRI_DIR} -> /data"
echo "FreeSurfer license mount: ${DOCKER_LICENSE_FILE} -> /fs_license/license.txt"
echo "Log file: ${LOG_FILE}"
echo "Provenance directory: ${PROVENANCE_DIR}"
echo

echo "Pulling FreeSurfer/FastSurfer Docker image if necessary."
docker_safe pull "${IMAGE}"

echo
build_python_image_if_needed

echo

echo "Recording run configuration and provenance."

{
    echo "Run timestamp: ${RUN_TIMESTAMP}"
    echo "Platform: ${PLATFORM}"
    echo "Subject ID: ${SUBJECT_ID}"
    echo "Docker image tag: ${IMAGE}"
    echo "Python skeletonization image: ${PYTHON_IMAGE}"
    echo "Skeleton dilation steps: ${SKELETON_DILATE_STEPS}"
    echo "Input labels: ${INPUT_LABELS}"
    echo "Final labels: ${FINAL_LABELS}"
    echo "FreeSurfer license file: ${LICENSE_FILE}"
    echo "Container license path: /fs_license/license.txt"
    echo
    echo "Host uname:"
    uname -a 2>&1 || true
} > "${PROVENANCE_DIR}/run_configuration.txt"

cp "${BASH_SOURCE[0]}" "${PROVENANCE_DIR}/script_used.sh"

if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${INPUT_LABELS}" > "${PROVENANCE_DIR}/input_sha256.txt"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${INPUT_LABELS}" > "${PROVENANCE_DIR}/input_sha256.txt"
else
    echo "SHA-256 utility not available." > "${PROVENANCE_DIR}/input_sha256.txt"
fi

docker_safe version > "${PROVENANCE_DIR}/docker_version.txt" 2>&1 || true
docker_safe image inspect "${IMAGE}" \
    > "${PROVENANCE_DIR}/docker_image_inspect.json" 2>&1 || true
docker_safe image inspect "${PYTHON_IMAGE}" \
    > "${PROVENANCE_DIR}/python_image_inspect.json" 2>&1 || true

echo

echo "Recording Python package versions."

docker_safe run \
    --rm \
    "${PYTHON_IMAGE}" \
    python - <<'PY' > "${PROVENANCE_DIR}/python_packages.txt" 2>&1 || true
import sys
print("Python:", sys.version)
for package in ["nibabel", "numpy", "skimage"]:
    try:
        mod = __import__(package)
        print(f"{package}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{package}: unavailable ({exc})")
PY

echo

echo "Step 1/3: preparing the whole-CC mask and NIfTI intermediate with FreeSurfer tools."

docker_safe run \
    --rm \
    "${DOCKER_USER_ARGS[@]}" \
    -e FS_LICENSE=/fs_license/license.txt \
    -e INPUT_NAME="${INPUT_NAME}" \
    -e INPUT_STEM="${INPUT_STEM}" \
    -v "${DOCKER_MRI_DIR}:/data" \
    -v "${DOCKER_LICENSE_FILE}:/fs_license/license.txt:ro" \
    --entrypoint /bin/bash \
    "${IMAGE}" \
    -lc '
        set -Eeuo pipefail

        INPUT_LABELS="/data/${INPUT_NAME}"
        SKELETON_DIR="/data/cc_skeleton"
        WHOLE_CC_MASK_MGZ="${SKELETON_DIR}/${INPUT_STEM}.whole.mask.mgz"
        WHOLE_CC_MASK_NII="${SKELETON_DIR}/${INPUT_STEM}.whole.mask.nii.gz"
        STATS_BEFORE="${SKELETON_DIR}/${INPUT_STEM}.stats.before.txt"

        mkdir -p "${SKELETON_DIR}"

        if [[ ! -f "${INPUT_LABELS}" ]]; then
            echo "Error: input segmentation not found: ${INPUT_LABELS}"
            exit 1
        fi

        echo "Merging labels 251..255 into one binary CC mask."
        mri_binarize \
            --i "${INPUT_LABELS}" \
            --match 251 252 253 254 255 \
            --o "${WHOLE_CC_MASK_MGZ}"

        echo "Converting the whole-CC binary mask to NIfTI."
        mri_convert "${WHOLE_CC_MASK_MGZ}" "${WHOLE_CC_MASK_NII}"

        echo "Recording pre-skeletonization voxel counts."
        mri_segstats \
            --seg "${INPUT_LABELS}" \
            --id 251 252 253 254 255 \
            --sum "${STATS_BEFORE}"

        echo "Created:"
        echo "  ${WHOLE_CC_MASK_MGZ}"
        echo "  ${WHOLE_CC_MASK_NII}"
        echo "  ${STATS_BEFORE}"
    '

echo

echo "Step 2/3: skeletonizing the whole-CC mask in 3D with scikit-image."
if [[ "${SKELETON_DILATE_STEPS}" -gt 0 ]]; then
    echo "The skeleton will be dilated by ${SKELETON_DILATE_STEPS} voxel(s) and clipped to the original CC mask."
else
    echo "The skeleton will not be dilated."
fi

docker_safe run \
    --rm \
    -i \
    "${DOCKER_USER_ARGS[@]}" \
    -e WHOLE_CC_MASK_NII="/data/cc_skeleton/${INPUT_STEM}.whole.mask.nii.gz" \
    -e SKELETON_BIN_NII="/data/cc_skeleton/${OUTPUT_STEM}.bin.nii.gz" \
    -e SKELETON_DILATE_STEPS="${SKELETON_DILATE_STEPS}" \
    -v "${DOCKER_MRI_DIR}:/data" \
    "${PYTHON_IMAGE}" \
    python - <<'PY'
import os
import sys

import nibabel as nib
import numpy as np
from skimage.morphology import ball, binary_dilation, skeletonize

mask_path = os.environ["WHOLE_CC_MASK_NII"]
out_path = os.environ["SKELETON_BIN_NII"]
dilate_steps = int(os.environ["SKELETON_DILATE_STEPS"])

print(f"Input mask: {mask_path}")
print(f"Output skeleton: {out_path}")
print(f"Skeleton dilation steps: {dilate_steps}")

img = nib.load(mask_path)
data = np.asanyarray(img.dataobj)
mask = data > 0

input_voxels = int(mask.sum())
print(f"Input CC voxels: {input_voxels}")

if input_voxels <= 0:
    print("Error: input CC mask is empty.", file=sys.stderr)
    sys.exit(1)

skeleton = skeletonize(mask, method="lee")
pure_skeleton_voxels = int(skeleton.sum())
print(f"Pure skeleton voxels: {pure_skeleton_voxels}")

if pure_skeleton_voxels <= 0:
    print("Error: skeleton is empty.", file=sys.stderr)
    sys.exit(1)

if dilate_steps > 0:
    print(f"Dilating skeleton with a spherical radius of {dilate_steps} voxel(s).")
    skeleton = binary_dilation(skeleton, footprint=ball(dilate_steps))
    dilated_voxels = int(skeleton.sum())
    print(f"Dilated skeleton voxels before CC clipping: {dilated_voxels}")

    skeleton = skeleton & mask
    clipped_voxels = int(skeleton.sum())
    print(f"Dilated skeleton voxels after CC clipping: {clipped_voxels}")

    if clipped_voxels <= 0:
        print("Error: thickened skeleton is empty after CC clipping.", file=sys.stderr)
        sys.exit(1)
else:
    skeleton = skeleton & mask
    clipped_voxels = int(skeleton.sum())
    print(f"Final skeleton voxels after CC clipping: {clipped_voxels}")

header = img.header.copy()
out_img = nib.Nifti1Image(skeleton.astype(np.uint8), img.affine, header)
out_img.set_data_dtype(np.uint8)
nib.save(out_img, out_path)

print("Skeletonization completed.")
PY

echo

echo "Step 3/3: converting skeleton back to MGZ and restoring labels 251..255."

docker_safe run \
    --rm \
    "${DOCKER_USER_ARGS[@]}" \
    -e FS_LICENSE=/fs_license/license.txt \
    -e INPUT_NAME="${INPUT_NAME}" \
    -e OUTPUT_NAME="${OUTPUT_NAME}" \
    -e OUTPUT_STEM="${OUTPUT_STEM}" \
    -v "${DOCKER_MRI_DIR}:/data" \
    -v "${DOCKER_LICENSE_FILE}:/fs_license/license.txt:ro" \
    --entrypoint /bin/bash \
    "${IMAGE}" \
    -lc '
        set -Eeuo pipefail

        INPUT_LABELS="/data/${INPUT_NAME}"
        FINAL_LABELS="/data/${OUTPUT_NAME}"
        SKELETON_DIR="/data/cc_skeleton"
        SKELETON_BIN_NII="${SKELETON_DIR}/${OUTPUT_STEM}.bin.nii.gz"
        SKELETON_BIN_MGZ="${SKELETON_DIR}/${OUTPUT_STEM}.bin.mgz"
        STATS_AFTER="${SKELETON_DIR}/${OUTPUT_STEM}.stats.after.txt"

        if [[ ! -f "${SKELETON_BIN_NII}" ]]; then
            echo "Error: skeleton NIfTI not found: ${SKELETON_BIN_NII}"
            exit 1
        fi

        echo "Converting skeleton NIfTI to MGZ."
        mri_convert "${SKELETON_BIN_NII}" "${SKELETON_BIN_MGZ}"

        echo "Applying skeleton mask to original multi-label segmentation."
        mri_mask \
            "${INPUT_LABELS}" \
            "${SKELETON_BIN_MGZ}" \
            "${FINAL_LABELS}"

        echo "Recording post-skeletonization voxel counts."
        mri_segstats \
            --seg "${FINAL_LABELS}" \
            --id 251 252 253 254 255 \
            --sum "${STATS_AFTER}"

        TOTAL_AFTER="$(
            awk '\''!/^#/ {sum += $3} END {print sum + 0}'\'' \
                "${STATS_AFTER}"
        )"

        if [[ "${TOTAL_AFTER}" -le 0 ]]; then
            echo "Error: skeletonized corpus callosum is empty."
            exit 1
        fi

        echo "Created:"
        echo "  ${SKELETON_BIN_NII}"
        echo "  ${SKELETON_BIN_MGZ}"
        echo "  ${FINAL_LABELS}"
        echo "  ${STATS_AFTER}"
    '

if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${FINAL_LABELS}" > "${PROVENANCE_DIR}/output_sha256.txt"
elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${FINAL_LABELS}" > "${PROVENANCE_DIR}/output_sha256.txt"
fi

cp "${STATS_BEFORE}" "${PROVENANCE_DIR}/$(basename "${STATS_BEFORE}")" 2>/dev/null || true
cp "${STATS_AFTER}" "${PROVENANCE_DIR}/$(basename "${STATS_AFTER}")" 2>/dev/null || true

echo

echo "Skeletonization results:"
echo "  Whole CC binary mask:"
echo "    ${WHOLE_CC_MASK_MGZ}"
echo "    ${WHOLE_CC_MASK_NII}"
echo
echo "  Binary skeleton:"
echo "    ${SKELETON_BIN_NII}"
echo "    ${SKELETON_BIN_MGZ}"
echo
echo "  Final multi-label skeleton segmentation:"
echo "    ${FINAL_LABELS}"
echo
echo "  Skeleton dilation steps:"
echo "    ${SKELETON_DILATE_STEPS}"
echo
echo "Use this file as the CC skeleton source in the DWI extraction pipeline:"
echo "  ${FINAL_LABELS}"
