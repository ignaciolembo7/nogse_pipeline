FastSurfer-CC corpus callosum workflow
======================================

This folder contains portable scripts for running FastSurfer-CC and deriving
corpus callosum masks in T1/FastSurfer space.

Expected structure
------------------

Place the scripts in the same directory that contains FASTSURFER_CC/.

Segmentations/
├── run_fastsurfer.sh
├── run_fastsurfer_erode.sh
├── run_fastsurfer_skeleton.sh
└── FASTSURFER_CC/
    ├── license.txt
    ├── IN/
    │   └── <T1 image>.nii.gz
    ├── OUT/
    │   └── <SUBJECT_ID>/
    │       └── mri/
    │           ├── callosum.CC.orig.mgz
    │           ├── callosum.CC.orig.erode1.mgz
    │           ├── callosum.CC.orig.sket.mgz
    │           └── callosum.CC.orig.sket1.mgz
    ├── LOGS/
    └── PROVENANCE/

1. Run FastSurfer-CC
--------------------

Example:

bash run_fastsurfer.sh sub-20220622_BRAIN sub-20220622_BRAIN.nii.gz 8

Main output:

FASTSURFER_CC/OUT/sub-20220622_BRAIN/mri/callosum.CC.orig.mgz

This file contains the corpus callosum labels 251..255.

2. Erode the whole corpus callosum mask, optional
------------------------------------------------

Example:

bash run_fastsurfer_erode.sh sub-20220622_BRAIN 1

Main output:

FASTSURFER_CC/OUT/sub-20220622_BRAIN/mri/callosum.CC.orig.erode1.mgz

The erosion script first merges labels 251..255 into one binary corpus callosum
mask, erodes only the outer boundary, and then restores the internal labels.
This avoids eroding the artificial borders between the five corpus callosum
subregions.

3. Skeletonize the whole corpus callosum mask
--------------------------------------------

MRtrix is not used for this step. In the available mrtrix3/mrtrix3 Docker image,
maskfilter supports filters such as clean, connect, dilate, erode, and median,
but not a thinning/skeletonization filter. The skeletonization script therefore
uses scikit-image for the actual 3D skeletonization, while FreeSurfer/FastSurfer
is still used for MGZ/NIfTI conversion and label restoration.

Usage:

bash run_fastsurfer_skeleton.sh SUBJECT_ID [SKELETON_DILATE_STEPS]

The second argument is optional:

0 = pure skeleton, default
1 = skeleton dilated by 1 voxel and clipped to the original CC mask
2 = skeleton dilated by 2 voxels and clipped to the original CC mask

Examples:

bash run_fastsurfer_skeleton.sh sub-20220622_BRAIN 0

bash run_fastsurfer_skeleton.sh sub-20220622_BRAIN 1

bash run_fastsurfer_skeleton.sh sub-20220622_BRAIN 2

Default outputs for INPUT_NAME=callosum.CC.orig.mgz:

SKELETON_DILATE_STEPS=0:
FASTSURFER_CC/OUT/sub-20220622_BRAIN/mri/callosum.CC.orig.sket.mgz

SKELETON_DILATE_STEPS=1:
FASTSURFER_CC/OUT/sub-20220622_BRAIN/mri/callosum.CC.orig.sket1.mgz

SKELETON_DILATE_STEPS=2:
FASTSURFER_CC/OUT/sub-20220622_BRAIN/mri/callosum.CC.orig.sket2.mgz

The script does the following:

1. Reads callosum.CC.orig.mgz.
2. Merges labels 251..255 into one whole-CC binary mask.
3. Converts the binary mask to NIfTI.
4. Skeletonizes the whole-CC mask in 3D with scikit-image.
5. Optionally dilates the skeleton by N voxels.
6. Clips the skeleton or thickened skeleton back to the original CC mask.
7. Converts the binary skeleton mask back to MGZ.
8. Applies the binary skeleton mask to the original multi-label segmentation.
9. Writes the final MGZ file with labels 251..255 preserved.

This means that a thickened skeleton never extends outside the original corpus
callosum segmentation:

thickened_skeleton = dilate(skeleton, N) ∩ original_CC_mask

The first run builds a local Docker image named:

cc-skeleton-python:latest

This image is based on python:3.11-slim and installs nibabel, numpy, and
scikit-image. Later runs reuse the image.

To force the local Python image to be rebuilt:

REBUILD_PYTHON_IMAGE=1 bash run_fastsurfer_skeleton.sh sub-20220622_BRAIN 1

To overwrite an existing skeleton output:

FORCE=1 bash run_fastsurfer_skeleton.sh sub-20220622_BRAIN 1

The same parameter can also be supplied as an environment variable:

SKELETON_DILATE_STEPS=1 FORCE=1 bash run_fastsurfer_skeleton.sh sub-20220622_BRAIN

4. Skeletonize an eroded corpus callosum mask, optional
------------------------------------------------------

To skeletonize the eroded file instead of the original file:

INPUT_NAME=callosum.CC.orig.erode1.mgz \
FORCE=1 \
bash run_fastsurfer_skeleton.sh sub-20220622_BRAIN 0

Main output:

FASTSURFER_CC/OUT/sub-20220622_BRAIN/mri/callosum.CC.orig.erode1.sket.mgz

To create a thickened skeleton from the eroded CC file:

INPUT_NAME=callosum.CC.orig.erode1.mgz \
FORCE=1 \
bash run_fastsurfer_skeleton.sh sub-20220622_BRAIN 1

Main output:

FASTSURFER_CC/OUT/sub-20220622_BRAIN/mri/callosum.CC.orig.erode1.sket1.mgz

You can still force a custom output name if needed:

INPUT_NAME=callosum.CC.orig.erode1.mgz \
OUTPUT_NAME=callosum.CC.orig.erode1.medialband1.mgz \
FORCE=1 \
bash run_fastsurfer_skeleton.sh sub-20220622_BRAIN 1

5. Outputs created by the skeletonization script
------------------------------------------------

Inside the subject mri folder:

FASTSURFER_CC/OUT/<SUBJECT_ID>/mri/callosum.CC.orig.sket.mgz
FASTSURFER_CC/OUT/<SUBJECT_ID>/mri/callosum.CC.orig.sket1.mgz
FASTSURFER_CC/OUT/<SUBJECT_ID>/mri/callosum.CC.orig.sket2.mgz

Inside the intermediate folder:

FASTSURFER_CC/OUT/<SUBJECT_ID>/mri/cc_skeleton/
├── callosum.CC.orig.whole.mask.mgz
├── callosum.CC.orig.whole.mask.nii.gz
├── callosum.CC.orig.sket.bin.nii.gz
├── callosum.CC.orig.sket.bin.mgz
├── callosum.CC.orig.sket1.bin.nii.gz
├── callosum.CC.orig.sket1.bin.mgz
├── callosum.CC.orig.stats.before.txt
├── callosum.CC.orig.sket.stats.after.txt
└── callosum.CC.orig.sket1.stats.after.txt

Only the intermediates corresponding to the command you ran are created.

Logs and provenance are saved under:

FASTSURFER_CC/LOGS/CC_SKELETON/
FASTSURFER_CC/PROVENANCE/CC_SKELETON/

6. Visual quality control
-------------------------

Open these files in MRIcroGL, Freeview, or FSLeyes:

orig_nu_brain.mgz
callosum.CC.orig.mgz
callosum.CC.orig.sket.mgz
callosum.CC.orig.sket1.mgz

The pure skeleton should lie inside the original corpus callosum segmentation
and retain labels 251, 252, 253, 254, and 255. It should not be created by
skeletonizing each subregion separately.

The thickened skeleton should also lie inside the original corpus callosum
segmentation. It should look like a medial band, not like the full original ROI.
For DWI-space extraction, start by comparing SKELETON_DILATE_STEPS=0 and
SKELETON_DILATE_STEPS=1 after registration with nearest-neighbor interpolation.
If SKELETON_DILATE_STEPS=1 avoids empty subregions in diffusion space, it is
usually the better compromise than using a pure one-voxel skeleton.
