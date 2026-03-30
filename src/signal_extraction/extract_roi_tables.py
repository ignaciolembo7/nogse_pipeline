from __future__ import annotations

"""
extract_roi_tables.py

This module complements coreg_extract.py.

It provides:
  - ROI dataclass (name + boolean mask)
  - Helpers to build ROI masks from:
      * a label image (multi-label atlas)
      * a CC label image (labels 251..255 etc.)
      * binary masks (0/1 or float masks)
  - MATLAB-like summary statistics across DWI volumes for each ROI
  - Excel writer with MATLAB-like sheet ordering

Design goals:
  - Be strict about *grid consistency* (same shape + compatible affine)
  - Be robust to NIfTI outputs that are accidentally saved as (X,Y,Z,1)
  - Keep memory usage reasonable (do not load full 4D into RAM)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import nibabel as nib


# ---------------------------------------------------------------------
# MATLAB-like statistics
# ---------------------------------------------------------------------
def matlab_std(x: np.ndarray) -> float:
    """
    MATLAB std(x) uses N-1 normalization (ddof=1).
    """
    return float(np.std(x, ddof=1)) if x.size > 1 else 0.0


def matlab_mad(x: np.ndarray) -> float:
    """
    Mean absolute deviation around the mean:
      mad(x, 1) in MATLAB  (flag=1 -> mean absolute deviation from mean)
    """
    if x.size == 0:
        return np.nan
    mu = float(np.mean(x))
    return float(np.mean(np.abs(x - mu)))


def matlab_mode(x: np.ndarray) -> float:
    """
    MATLAB-like mode:
      - For ties, MATLAB returns the smallest value.
    Notes:
      - For continuous float-valued DWIs, the mode is often not very informative
        because many values are unique. We keep it for compatibility.
    """
    if x.size == 0:
        return np.nan
    vals, counts = np.unique(x, return_counts=True)
    m = counts.max()
    return float(vals[counts == m].min())

def quantized_mode(x: np.ndarray, *, step: float = 1.0) -> float:
    """
    Quantized mode for continuous-valued intensities.

    We quantize values to multiples of `step` and then compute the mode.
    Tie-breaking: smallest value among ties (MATLAB-like).
    """
    if x.size == 0:
        return np.nan
    if step <= 0:
        raise ValueError("step must be > 0")

    x = np.asarray(x, dtype=np.float64)
    q = np.rint(x / step).astype(np.int64)  # quantized integers

    vals, counts = np.unique(q, return_counts=True)
    m = counts.max()
    v = vals[counts == m].min()
    return float(v * step)


# ---------------------------------------------------------------------
# ROI container
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ROI:
    """
    ROI representation used by the pipeline.

    Attributes
    ----------
    name : str
        ROI name used as column header in output tables.
    mask : np.ndarray (bool)
        3D boolean mask in the *same grid* as the DWI spatial grid.
    """
    name: str
    mask: np.ndarray  # bool 3D


# ---------------------------------------------------------------------
# NIfTI helpers
# ---------------------------------------------------------------------
def _load_nii(p: Path) -> nib.Nifti1Image:
    """Load a NIfTI image from disk."""
    return nib.load(str(p))


def _as_3d(dataobj, what: str) -> np.ndarray:
    """
    Convert a nibabel dataobj into a strict 3D array.
    Accepts:
      - 2D single-slice data stored as (X,Y), promoted to (X,Y,1)
      - 3D data stored as (X,Y,Z)
      - 4D singleton data stored as (X,Y,Z,1), squeezed to (X,Y,Z)
    """
    a = np.asanyarray(dataobj)

    # Promote 2D single-slice images to a 3D volume so the rest of the
    # pipeline can treat single-slice phantoms like standard 3D data.
    if a.ndim == 2:
        a = a[..., np.newaxis]

    # Only squeeze a singleton 4th dim (common pitfall from some tools)
    if a.ndim == 4 and a.shape[-1] == 1:
        a = a[..., 0]

    if a.ndim != 3:
        raise ValueError(f"{what} must be 3D (or 4D with T=1). Got shape={a.shape}")

    return a


def _ensure_same_grid(
    dwi_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    what: str,
    *,
    check_affine: bool = True,
    affine_rtol: float = 1e-4,
    affine_atol: float = 1e-3,
) -> None:
    """
    Ensure mask image matches the DWI spatial grid.

    Minimum requirement: same spatial shape (X,Y,Z).
    Recommended: also require compatible affine (same orientation/voxel-to-world).
    """
    dwi_shape = tuple(dwi_img.shape[:3])
    mask_shape = tuple(_as_3d(mask_img.dataobj, what).shape)

    if dwi_shape != mask_shape:
        raise ValueError(
            f"{what}: shape {mask_shape} != DWI spatial {dwi_shape}. "
            "The mask must be resampled into the SAME DWI grid."
        )

    if check_affine:
        if not np.allclose(dwi_img.affine, mask_img.affine, rtol=affine_rtol, atol=affine_atol):
            raise ValueError(
                f"{what}: affine does not match DWI affine (within tolerances). "
                "Shapes match but the world geometry differs. "
                "Please resample the mask into the exact DWI reference."
            )


def _mask_to_bool(a3d: np.ndarray) -> np.ndarray:
    """
    Convert a 3D array into a boolean mask.
    Uses >0.5 threshold to be robust to float masks (e.g. resampling artifacts).
    """
    if a3d.dtype == np.bool_:
        return a3d
    return a3d > 0.5


# ---------------------------------------------------------------------
# ROI builders
# ---------------------------------------------------------------------
def build_rois_from_labelmask(
    labelmask_path: Path,
    dwi_img: nib.Nifti1Image,
    labels: list[tuple[str, int]],
) -> list[ROI]:
    """
    Build a list of ROIs from a *label image* in DWI space.

    Parameters
    ----------
    labelmask_path : Path
        Path to a label image where voxels contain integer labels (0=background).
    dwi_img : nib.Nifti1Image
        Loaded DWI image (used to check grid).
    labels : list[(name, label_id)]
        Each entry defines a ROI by integer label_id.

    Returns
    -------
    list[ROI]
    """
    lm_img = _load_nii(labelmask_path)
    _ensure_same_grid(dwi_img, lm_img, "Label mask")

    lm = _as_3d(lm_img.dataobj, "Label mask").astype(np.int32)
    return [ROI(name=name, mask=(lm == int(lab))) for name, lab in labels]


def build_rois_from_cc_mask(
    cc_mask_path: Path,
    dwi_img: nib.Nifti1Image,
    cc_names: list[str],
) -> list[ROI]:
    """
    Build ROIs from a CC label image (nonzero labels).
    Labels are read from the volume (unique nonzero values), sorted ascending,
    and assigned to cc_names in that order.

    This matches typical CC labels 251..255, but can be used generically if
    your file has K distinct nonzero labels.

    Raises if the number of labels != len(cc_names).
    """
    cc_img = _load_nii(cc_mask_path)
    _ensure_same_grid(dwi_img, cc_img, "CC mask")

    cc = _as_3d(cc_img.dataobj, "CC mask").astype(np.int32)
    labs = sorted(int(v) for v in np.unique(cc) if v != 0)

    if len(labs) != len(cc_names):
        raise ValueError(
            f"CC mask has {len(labs)} nonzero labels but cc_names has {len(cc_names)} entries."
        )

    return [ROI(name=name, mask=(cc == lab)) for lab, name in zip(labs, cc_names)]


def build_roi_from_binary_mask(
    mask_path: Path,
    dwi_img: nib.Nifti1Image,
    name: str,
) -> ROI:
    """
    Build a single ROI from a binary mask in DWI space.

    Notes:
      - Accepts masks stored as uint8 0/1, but also float masks.
      - Uses threshold > 0.5, not equality == 1, to be robust.
    """
    m_img = _load_nii(mask_path)
    _ensure_same_grid(dwi_img, m_img, f"Mask {name}")

    m = _as_3d(m_img.dataobj, f"Mask {name}")
    return ROI(name=name, mask=_mask_to_bool(m))


# ---------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------
def _read_bvals(bval_path: Path, nvol: int) -> np.ndarray:
    """
    Read b-values from a .bval file and validate length.
    FSL bval files are usually 1xN, but we robustly flatten.
    """
    bval = np.loadtxt(bval_path, dtype=float)
    bval = np.asarray(bval).reshape(-1).astype(float)

    if bval.size != nvol:
        raise ValueError(f"bval size {bval.size} != nvol {nvol} for file: {bval_path}")

    return bval


def extract_tables(
    dwi_path: Path,
    bval_path: Path,
    rois: list[ROI],
) -> dict[str, pd.DataFrame]:
    """
    Fast table extraction:
      - loads each DWI volume only once
      - computes all ROI stats for that volume

    Output keys: mean, std, median, mad, mode
    """
    dwi_img = _load_nii(dwi_path)
    if len(dwi_img.shape) != 4:
        raise ValueError(f"DWI must be 4D (X,Y,Z,Nvol). Got shape={dwi_img.shape}: {dwi_path}")

    nvol = int(dwi_img.shape[3])
    bval = _read_bvals(bval_path, nvol)

    spatial_shape = dwi_img.shape[:3]

    # Precompute ROI flat indices once
    roi_names: list[str] = []
    roi_idx: list[np.ndarray] = []
    for roi in rois:
        if roi.mask.shape != spatial_shape:
            raise ValueError(
                f"ROI '{roi.name}' mask shape {roi.mask.shape} != DWI spatial shape {spatial_shape}."
            )
        idx = np.flatnonzero(roi.mask.reshape(-1))
        if idx.size == 0:
            raise ValueError(f"ROI '{roi.name}' is empty.")
        roi_names.append(roi.name)
        roi_idx.append(idx)

    # Allocate result arrays: dict[stat][roi] -> (nvol,)
    mean_arr = {name: np.empty(nvol, dtype=float) for name in roi_names}
    std_arr  = {name: np.empty(nvol, dtype=float) for name in roi_names}
    med_arr  = {name: np.empty(nvol, dtype=float) for name in roi_names}
    mad_arr  = {name: np.empty(nvol, dtype=float) for name in roi_names}
    mode_arr = {name: np.empty(nvol, dtype=float) for name in roi_names}

    dataobj = dwi_img.dataobj  # lazy proxy

    for j in range(nvol):
        # Load volume ONCE
        vol = np.asanyarray(dataobj[..., j], dtype=np.float64).reshape(-1)

        for name, idx in zip(roi_names, roi_idx):
            x = vol[idx]

            mean_arr[name][j] = float(np.mean(x))
            std_arr[name][j]  = matlab_std(x)
            med_arr[name][j]  = float(np.median(x))
            mad_arr[name][j]  = matlab_mad(x)

            # Quantized mode (recommended for MRI intensities)
            mode_arr[name][j] = quantized_mode(x, step=1.0)

    # Build output tables
    out = {
        "mean":   pd.DataFrame({"bvalues": bval}),
        "std":    pd.DataFrame({"bvalues": bval}),
        "median": pd.DataFrame({"bvalues": bval}),
        "mad":    pd.DataFrame({"bvalues": bval}),
        "mode":   pd.DataFrame({"bvalues": bval}),
    }

    for name in roi_names:
        out["mean"][name]   = mean_arr[name]
        out["std"][name]    = std_arr[name]
        out["median"][name] = med_arr[name]
        out["mad"][name]    = mad_arr[name]
        out["mode"][name]   = mode_arr[name]

    return out


# ---------------------------------------------------------------------
# Excel writer (MATLAB-like ordering)
# ---------------------------------------------------------------------
def write_excel_like_matlab(tables: dict[str, pd.DataFrame], out_xlsx: Path) -> None:
    """
    Write the tables dict to an .xlsx with a fixed sheet order, matching your MATLAB output.

    Sheets:
      Sheet1 -> mean
      Sheet2 -> std
      Sheet3 -> median
      Sheet4 -> mad
      Sheet5 -> mode
    """
    order = [("Sheet1", "mean"), ("Sheet2", "std"), ("Sheet3", "median"), ("Sheet4", "mad"), ("Sheet5", "mode")]
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        for sheet, key in order:
            if key not in tables:
                raise KeyError(f"Missing key '{key}' in tables dict. Available keys: {list(tables.keys())}")
            tables[key].to_excel(w, sheet_name=sheet, index=False)
