from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Tuple

try:
    import repo_bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import repo_bootstrap  # noqa: F401

import nibabel as nib
import numpy as np

from extract_roi_tables import (
    build_roi_from_binary_mask,
    extract_tables,
    write_excel_like_matlab,
)

"""
coreg_extract_brain.py

Brain-oriented pipeline:
1) Compute b0_mean from DWI
2) BET skull-strip b0_mean
3) Register b0_brain -> T1_brain with ANTs
4) Warp selected atlas labels from T1 to DWI
5) Optionally warp syringe mask from T1 to DWI
6) Build ROI binary masks and a single multi-label image
7) Extract ROI tables and write Excel
8) Also warp T1 and T1_brain to DWI space
"""


# ---------------------------------------------------------------------
# Shell / I/O utilities
# ---------------------------------------------------------------------
def run(cmd: list[str], dry_run: bool = False) -> None:
    """Print and run a shell command."""
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def maybe_reuse_existing(path: Path, what: str, fail_on_existing: bool) -> bool:
    """
    Return True if `path` already exists and should be reused.

    If --fail-on-existing is set, raise SystemExit instead.
    """
    if path.exists():
        if fail_on_existing:
            raise SystemExit(f"[STOP] Found existing {what} and --fail-on-existing was set: {path}")
        print(f"[INFO] Reusing existing {what}: {path}")
        return True
    return False


def read_nonnegative_int_env(name: str, default: int = 0) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"[STOP] {name} must be a non-negative integer. Got {raw!r}.") from exc
    if value < 0:
        raise SystemExit(f"[STOP] {name} must be a non-negative integer. Got {value}.")
    return value


def read_bool_env(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise SystemExit(f"[STOP] {name} must be true/false or 1/0. Got {raw!r}.")


def resolve_existing_path(raw: Path, *roots: Path) -> Path:
    """
    Resolve `raw` against a list of roots if it is relative.
    Returns the first existing candidate if found, else the first candidate path.
    """
    if raw.is_absolute():
        return raw

    for root in roots:
        cand = root / raw
        if cand.exists():
            return cand

    return (roots[0] / raw) if roots else raw


def strip_nii_ext(p: Path) -> str:
    """Return filename without .nii or .nii.gz extension."""
    n = p.name
    if n.endswith(".nii.gz"):
        return n[:-7]
    if n.endswith(".nii"):
        return n[:-4]
    return p.stem


def dcm2niix_variant(seq_name: str) -> str:
    """Return dcm2niix conflict suffix: "", "a", "b", etc."""
    m = re.search(r"_(\d+)([a-z])$", seq_name, flags=re.IGNORECASE)
    return m.group(2).lower() if m else ""


def dwi_variant_matches(seq_name: str, wanted: str) -> bool:
    """Filter dcm2niix variants. Use 'none', 'a', 'b', ... or 'all'."""
    wanted_norm = wanted.strip().lower()
    if wanted_norm in {"all", "*", "any"}:
        return True
    variant = dcm2niix_variant(seq_name)
    if wanted_norm in {"", "none", "primary", "base"}:
        return variant == ""
    return variant == wanted_norm


def cut_before_any(s: str, tokens: Iterable[str]) -> Tuple[str, str | None]:
    """
    Cut `s` at the first occurrence of any token in `tokens`.
    Returns (prefix, token_used). If no token is found, returns (s, None).
    """
    best_idx = None
    best_tok = None
    for tok in tokens:
        if not tok:
            continue
        idx = s.find(tok)
        if idx != -1 and (best_idx is None or idx < best_idx):
            best_idx = idx
            best_tok = tok
    if best_idx is None:
        return s, None
    return s[:best_idx], best_tok


def sanitize_name(s: str) -> str:
    """Make a safe filename-like token (letters/numbers/_/- only)."""
    s = s.strip()
    s = re.sub(r"[^\w\-]+", "_", s)
    return s.strip("_")


def load_nifti_3d(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """
    Load a NIfTI and return (img, data_3d).

    Supports:
      - 3D: (X,Y,Z)
      - 4D singleton: (X,Y,Z,1) -> squeezed to (X,Y,Z)

    Raises SystemExit for any other shape.
    """
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)

    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]

    if data.ndim != 3:
        raise SystemExit(f"[STOP] Expected 3D (or 4D with T=1) but got shape={data.shape}: {path}")

    return img, data


def count_nonzero_mask_voxels(mask_path: Path) -> int:
    """Count nonzero voxels in a (3D or 4D singleton) mask."""
    _img, data = load_nifti_3d(mask_path)
    return int(np.count_nonzero(data))


# ---------------------------------------------------------------------
# DWI discovery
# ---------------------------------------------------------------------
def find_dwis(
    exp_root: Path,
    grad_root: Path,
    dwi_glob: str,
    cut_tokens: list[str],
    dwi_variant: str,
) -> list[tuple[Path, Path, Path, str, str, str]]:
    """
    Find DWI NIfTIs and matching gradient sidecars.

    Returns a list of tuples:
      (dwi_nii, grad_values, grad_vectors, grad_kind, seq_name_full, seq_no_ext)

    Where seq_no_ext is computed by cutting seq_name_full at the earliest cut_token.
    """
    dwis: list[tuple[Path, Path, Path, str, str, str]] = []
    for nii in sorted(exp_root.glob(dwi_glob)):
        seq_name_full = strip_nii_ext(nii)
        if not dwi_variant_matches(seq_name_full, dwi_variant):
            continue
        seq_no_ext, _tok = cut_before_any(seq_name_full, cut_tokens)

        bval = grad_root / f"{seq_no_ext}.bval"
        bvec = grad_root / f"{seq_no_ext}.bvec"
        if bval.exists() and bvec.exists():
            dwis.append((nii, bval, bvec, "b", seq_name_full, seq_no_ext))
            continue

        gval = grad_root / f"{seq_no_ext}.gval"
        gvec = grad_root / f"{seq_no_ext}.gvec"
        if gval.exists() and gvec.exists():
            dwis.append((nii, gval, gvec, "g", seq_name_full, seq_no_ext))

    if not dwis:
        raise SystemExit(
            f"[STOP] No DWIs found with glob '{dwi_glob}' in {exp_root} "
            f"and dwi_variant='{dwi_variant}' with gradients in {grad_root}. "
            "Expected .bval/.bvec or .gval/.gvec files."
        )

    return dwis


# ---------------------------------------------------------------------
# FreeSurfer export (once per subject)
# ---------------------------------------------------------------------
def prep_struct_once(
    fs_subj: Path,
    out_root: Path,
    dry_run: bool = False,
    fail_on_existing: bool = False,
) -> tuple[Path, Path, Path]:
    """
    Prepare structural files in out_root:

      - T1.nii.gz
      - T1_brain.nii.gz
      - wmparc.nii.gz

    Logic:
      1) If the NIfTI files already exist in out_root, reuse them.
      2) For any missing file, try to create it from FreeSurfer .mgz.
      3) Only fail if, after that, some required output is still missing.
    """
    out_root.mkdir(parents=True, exist_ok=True)

    t1_full = out_root / "T1.nii.gz"
    t1_brain = out_root / "T1_brain.nii.gz"
    wmparc = out_root / "wmparc.nii.gz"

    maybe_reuse_existing(t1_full, "T1.nii.gz", fail_on_existing)
    maybe_reuse_existing(t1_brain, "T1_brain.nii.gz", fail_on_existing)
    maybe_reuse_existing(wmparc, "wmparc.nii.gz", fail_on_existing)

    t1_mgz = fs_subj / "mri" / "T1.mgz"
    brain_mgz = fs_subj / "mri" / "brain.mgz"
    wmparc_mgz = fs_subj / "mri" / "wmparc.mgz"

    sources = [
        (t1_full, t1_mgz, "T1"),
        (t1_brain, brain_mgz, "T1_brain"),
        (wmparc, wmparc_mgz, "wmparc"),
    ]

    for out_nii, src_mgz, label in sources:
        if out_nii.exists():
            continue

        if not src_mgz.exists():
            print(f"[WARN] Missing FreeSurfer source for {label}: {src_mgz}")
            continue

        run(
            ["mri_convert", "-it", "mgz", "-ot", "nii", str(src_mgz), str(out_nii)],
            dry_run=dry_run,
        )

    missing_outputs = [str(p) for p in (t1_full, t1_brain, wmparc) if not p.exists()]
    if missing_outputs:
        raise SystemExit(
            "[STOP] Could not prepare structural files.\n"
            "The following required NIfTI files are still missing:\n"
            + "\n".join(f"  {p}" for p in missing_outputs)
            + "\n\n"
            "If you want to reuse already exported files, place them exactly here with these names:\n"
            f"  {t1_full}\n"
            f"  {t1_brain}\n"
            f"  {wmparc}\n"
            "\n"
            "Otherwise the script expects the corresponding FreeSurfer files here:\n"
            f"  {t1_mgz}\n"
            f"  {brain_mgz}\n"
            f"  {wmparc_mgz}"
        )

    return t1_full, t1_brain, wmparc


# ---------------------------------------------------------------------
# b0 + BET
# ---------------------------------------------------------------------
def make_b0_mean_with_mrtrix_fsl(
    dwi_nii: Path,
    bvec: Path,
    bval: Path,
    out_seq: Path,
    reuse_reference: bool = True,
    dry_run: bool = False,
    fail_on_existing: bool = False,
) -> Path:
    """
    Compute mean b0:
      - dwiextract -bzero (MRtrix) -> NII_b0.nii.gz
      - fslmaths -Tmean            -> b0_mean.nii.gz
    """
    out_seq.mkdir(parents=True, exist_ok=True)
    nii_b0 = out_seq / "NII_b0.nii.gz"
    b0_mean = out_seq / "b0_mean.nii.gz"

    if reuse_reference and maybe_reuse_existing(b0_mean, "b0_mean.nii.gz", fail_on_existing):
        return b0_mean
    if not reuse_reference and (nii_b0.exists() or b0_mean.exists()):
        print(f"[INFO] Overwriting reference images: {nii_b0}, {b0_mean}")

    run(
        ["dwiextract", "-bzero", str(dwi_nii), str(nii_b0), "--fslgrad", str(bvec), str(bval), "-force"],
        dry_run=dry_run,
    )
    run(["fslmaths", str(nii_b0), "-Tmean", str(out_seq / "b0_mean")], dry_run=dry_run)
    return b0_mean


def make_full_mean_with_fsl(
    dwi_nii: Path,
    out_seq: Path,
    dummy_scans: int = 0,
    reuse_reference: bool = True,
    dry_run: bool = False,
    fail_on_existing: bool = False,
) -> Path:
    """
    Compute the mean across DWI volumes:
      - write the selected 4D DWI volumes -> NII_mean.nii.gz
      - fslmaths -Tmean                   -> mean.nii.gz
    """
    if dummy_scans < 0:
        raise SystemExit(f"[STOP] --dummy-scans must be >= 0. Got {dummy_scans}.")

    dwi_shape = nib.load(str(dwi_nii)).shape
    if len(dwi_shape) != 4:
        raise SystemExit(f"[STOP] DWI must be 4D. Got shape={dwi_shape}: {dwi_nii}")
    nvol = int(dwi_shape[3])
    if dummy_scans >= nvol:
        raise SystemExit(
            f"[STOP] --dummy-scans must be smaller than the number of DWI volumes. "
            f"Got --dummy-scans={dummy_scans}, nvol={nvol}: {dwi_nii}"
        )

    out_seq.mkdir(parents=True, exist_ok=True)
    nii_mean = out_seq / "NII_mean.nii.gz"
    mean_img = out_seq / "mean.nii.gz"

    if reuse_reference and maybe_reuse_existing(mean_img, "mean.nii.gz", fail_on_existing):
        return mean_img
    if not reuse_reference and (nii_mean.exists() or mean_img.exists()):
        print(f"[INFO] Overwriting reference images: {nii_mean}, {mean_img}")

    if dummy_scans > 0:
        run(["fslroi", str(dwi_nii), str(nii_mean), str(dummy_scans), "-1"], dry_run=dry_run)
    elif dry_run:
        print(f"cp {dwi_nii} {nii_mean}")
    else:
        shutil.copyfile(dwi_nii, nii_mean)

    run(["fslmaths", str(nii_mean), "-Tmean", str(out_seq / "mean")], dry_run=dry_run)
    return mean_img


def make_zero_gradient_mean_from_values(
    dwi_nii: Path,
    grad_values_path: Path,
    out_seq: Path,
    reuse_reference: bool = True,
    dry_run: bool = False,
    fail_on_existing: bool = False,
) -> Path:
    """
    Compute a mean reference from all zero-gradient volumes using a .bval or .gval file.
    """
    out_seq.mkdir(parents=True, exist_ok=True)
    nii_b0 = out_seq / "NII_b0.nii.gz"
    b0_mean = out_seq / "b0_mean.nii.gz"

    if reuse_reference and maybe_reuse_existing(b0_mean, "b0_mean.nii.gz", fail_on_existing):
        return b0_mean
    if not reuse_reference and (nii_b0.exists() or b0_mean.exists()):
        print(f"[INFO] Overwriting reference images: {nii_b0}, {b0_mean}")

    values = np.loadtxt(grad_values_path, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    zero_idx = np.flatnonzero(np.isclose(values, 0.0, rtol=0.0, atol=1e-9))
    if zero_idx.size == 0:
        raise SystemExit(
            f"[STOP] Could not find zero-gradient volumes in {grad_values_path}. "
            "Use --mean if this sequence should be collapsed across all volumes."
        )

    if dry_run:
        print(
            f"[DRY] Would extract {zero_idx.size} zero-gradient volumes from "
            f"{dwi_nii} using {grad_values_path} -> {nii_b0}"
        )
    else:
        dwi_img = nib.load(str(dwi_nii))
        data = np.asanyarray(dwi_img.dataobj)[..., zero_idx]
        out_img = nib.Nifti1Image(data, dwi_img.affine, dwi_img.header.copy())
        nib.save(out_img, str(nii_b0))

    run(["fslmaths", str(nii_b0), "-Tmean", str(out_seq / "b0_mean")], dry_run=dry_run)
    return b0_mean


def bet_b0(
    b0_mean: Path,
    out_seq: Path,
    seq_name_full: str,
    bet_f: float,
    dry_run: bool = False,
    fail_on_existing: bool = False,
) -> Path:
    """
    Skull-strip b0_mean using FSL BET.

    Output prefix:
      <out_seq>/<seq_name_full>_b0_brain

    Returns the path to:
      <prefix>.nii.gz
    """
    out_prefix = out_seq / f"{seq_name_full}_b0_brain"
    out_nii = Path(str(out_prefix) + ".nii.gz")

    if maybe_reuse_existing(out_nii, f"{seq_name_full}_b0_brain.nii.gz", fail_on_existing):
        return out_nii

    run(["bet", str(b0_mean), str(out_prefix), "-m", "-f", str(bet_f), "-R"], dry_run=dry_run)
    return out_nii


# ---------------------------------------------------------------------
# ANTs registration + applying transforms
# ---------------------------------------------------------------------
def ants_register_b0_to_t1(
    t1_brain: Path,
    b0_brain: Path,
    out_prefix: Path,
    dry_run: bool = False,
    fail_on_existing: bool = False,
) -> Path:
    """
    Register b0_brain (moving) to t1_brain (fixed) using Rigid + Affine transforms.
    Returns path to the affine matrix: <out_prefix>0GenericAffine.mat
    """
    affine = out_prefix.parent / (out_prefix.name + "0GenericAffine.mat")
    if maybe_reuse_existing(affine, f"{out_prefix.name}0GenericAffine.mat", fail_on_existing):
        return affine

    cmd = [
        "antsRegistration",
        "--dimensionality", "3",
        "--float", "0",
        "--output", f"[{out_prefix},{out_prefix}.nii.gz,{out_prefix}_inv.nii.gz]",
        "--interpolation", "Linear",
        "--use-histogram-matching", "1",
        "--initial-moving-transform", f"[{t1_brain},{b0_brain},1]",
        "--transform", "Rigid[0.1]",
        "--metric", f"MI[{t1_brain},{b0_brain},1,32,Regular,0.25]",
        "--convergence", "[1000x500x250x100,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
        "--transform", "Affine[0.1]",
        "--metric", f"MI[{t1_brain},{b0_brain},1,32,Regular,0.25]",
        "--convergence", "[1000x500x250x100,1e-6,10]",
        "--shrink-factors", "8x4x2x1",
        "--smoothing-sigmas", "3x2x1x0vox",
    ]
    run(cmd, dry_run=dry_run)
    return affine


def ants_apply_inverse_label(
    moving_label_t1: Path,
    ref_dwi: Path,
    affine_b0_to_t1: Path,
    out_label_dwi: Path,
    dry_run: bool = False,
    fail_on_existing: bool = False,
) -> None:
    """
    Warp a label image from T1 space to DWI space using the inverse of affine_b0_to_t1.
    Uses NearestNeighbor interpolation.
    """
    out_label_dwi.parent.mkdir(parents=True, exist_ok=True)
    if maybe_reuse_existing(out_label_dwi, out_label_dwi.name, fail_on_existing):
        return

    run(
        [
            "antsApplyTransforms",
            "-e", "0",
            "-i", str(moving_label_t1),
            "-r", str(ref_dwi),
            "-o", str(out_label_dwi),
            "-t", f"[{affine_b0_to_t1},1]",
            "-n", "NearestNeighbor",
            "-v", "1",
        ],
        dry_run=dry_run,
    )


def ants_apply_inverse_image(
    moving_img_t1: Path,
    ref_dwi: Path,
    affine_b0_to_t1: Path,
    out_img_dwi: Path,
    *,
    interp: str = "Linear",
    dry_run: bool = False,
    fail_on_existing: bool = False,
) -> None:
    """
    Warp an intensity image from T1 space to DWI space using the inverse of affine_b0_to_t1.
    Uses Linear interpolation by default.
    """
    out_img_dwi.parent.mkdir(parents=True, exist_ok=True)
    if maybe_reuse_existing(out_img_dwi, out_img_dwi.name, fail_on_existing):
        return

    run(
        [
            "antsApplyTransforms",
            "-e", "0",
            "-i", str(moving_img_t1),
            "-r", str(ref_dwi),
            "-o", str(out_img_dwi),
            "-t", f"[{affine_b0_to_t1},1]",
            "-n", interp,
            "-v", "1",
        ],
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------
# ROI parsing and creation
# ---------------------------------------------------------------------
def fslmeants_mask(dwi: Path, mask: Path, out_txt: Path, dry_run: bool = False) -> None:
    """Write fslmeants timecourse for a mask."""
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    run(["fslmeants", "-i", str(dwi), "-o", str(out_txt), "-m", str(mask)], dry_run=dry_run)


def parse_atlas_roi_spec(spec: str) -> tuple[int, str]:
    """
    Parse CLI ROI spec "LABEL:Name", e.g.:
      4:Left-Lateral-Ventricle
      43:Right-Lateral-Ventricle
      251:PostCC
    """
    if ":" not in spec:
        raise SystemExit(
            f"[STOP] Invalid --atlas-roi '{spec}'. Use LABEL:Name, e.g. 4:Left-Lateral-Ventricle"
        )
    label_str, name = spec.split(":", 1)
    label_str = label_str.strip()
    name = name.strip()

    try:
        label_value = int(label_str)
    except ValueError as e:
        raise SystemExit(f"[STOP] Invalid --atlas-roi '{spec}'. LABEL must be an integer.") from e

    if not name:
        raise SystemExit(f"[STOP] Invalid --atlas-roi '{spec}'. Missing Name after ':'.")

    return label_value, name


def dedupe_label_specs(label_specs: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """
    Deduplicate label specs by label id, keeping the first occurrence.
    """
    seen: set[int] = set()
    out: list[tuple[int, str]] = []
    for lv, nm in label_specs:
        if lv in seen:
            continue
        seen.add(lv)
        out.append((lv, nm))
    return out


def write_binary_mask_from_label_image(
    label_img_path: Path,
    label_value: int,
    out_mask: Path,
) -> int:
    """
    Create a binary mask for a given label_value from a label image in DWI space.
    Supports 3D or 4D singleton inputs.
    Returns the number of nonzero voxels.
    """
    img, data = load_nifti_3d(label_img_path)
    data_i = np.rint(data).astype(np.int32)

    mask = (data_i == int(label_value)).astype(np.uint8)
    nvox = int(mask.sum())

    out_mask.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(mask, img.affine, img.header.copy())
    out_img.set_data_dtype(np.uint8)
    nib.save(out_img, str(out_mask))

    return nvox


def write_selected_label_image(
    label_img_path: Path,
    label_specs: list[tuple[int, str]],
    out_img_path: Path,
) -> int:
    """
    Create a label image in the same space as label_img_path, but keeping only labels in label_specs.
    All other voxels are set to 0.
    """
    img, data = load_nifti_3d(label_img_path)
    data_i = np.rint(data).astype(np.int32)

    out = np.zeros(data_i.shape, dtype=np.int16)
    for label_value, _roi_name in label_specs:
        out[data_i == int(label_value)] = int(label_value)

    nvox = int(np.count_nonzero(out))

    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(out, img.affine, img.header.copy())
    out_img.set_data_dtype(np.int16)
    nib.save(out_img, str(out_img_path))

    return nvox


def build_all_rois_multilabel(
    ref_img_path: Path,
    atlas_label_img_path: Path | None,
    atlas_specs: list[tuple[int, str]],
    syringe_mask_dwi: Path | None,
    manual_specs: list[tuple[str, Path]],
    out_img_path: Path,
    out_csv_path: Path,
    *,
    fail_on_existing: bool,
) -> None:
    """
    Build a single 3D multi-label NIfTI in DWI space with:

      - Atlas labels: keep original label values
      - Syringe mask: label 9000
      - Manual masks: labels 9100, 9101, ...

    Also writes a CSV mapping {label_id -> name}.
    """
    nii_exists = out_img_path.exists()
    csv_exists = out_csv_path.exists()

    if nii_exists and fail_on_existing:
        raise SystemExit(f"[STOP] Found existing {out_img_path} and --fail-on-existing was set.")
    if csv_exists and fail_on_existing:
        raise SystemExit(f"[STOP] Found existing {out_csv_path} and --fail-on-existing was set.")

    if nii_exists and csv_exists:
        print(f"[INFO] Reusing existing ALL_ROIS multilabel: {out_img_path}")
        print(f"[INFO] Reusing existing mapping CSV:        {out_csv_path}")
        return

    if nii_exists and not csv_exists:
        print("[INFO] ALL_ROIS NIfTI exists but mapping CSV is missing; rebuilding both for consistency.")
        out_img_path.unlink()

    ref_img, _ = load_nifti_3d(ref_img_path)
    ref_shape = ref_img.shape[:3]
    labels = np.zeros(ref_shape, dtype=np.int16)

    mapping_rows: list[tuple[int, str]] = []

    if atlas_label_img_path is not None and atlas_specs:
        _aimg, adata = load_nifti_3d(atlas_label_img_path)
        adata = np.rint(adata).astype(np.int32)

        if adata.shape[:3] != ref_shape:
            raise SystemExit(
                f"[STOP] Shape mismatch: atlas_in_dwi {adata.shape[:3]} != ref {ref_shape}: {atlas_label_img_path}"
            )

        keep = np.zeros(ref_shape, dtype=bool)
        for lv, name in atlas_specs:
            keep |= (adata == int(lv))
            mapping_rows.append((int(lv), str(name)))

        labels[keep] = adata[keep].astype(np.int16)

    if syringe_mask_dwi is not None and syringe_mask_dwi.exists():
        syr_label = 9000
        _simg, sdata = load_nifti_3d(syringe_mask_dwi)

        if sdata.shape[:3] != ref_shape:
            raise SystemExit(
                f"[STOP] Shape mismatch: syringe {sdata.shape[:3]} != ref {ref_shape}: {syringe_mask_dwi}"
            )

        mask = sdata > 0.5
        new_vox = mask & (labels == 0)
        labels[new_vox] = syr_label
        mapping_rows.append((syr_label, "Syringe"))

    manual_label_start = 9100
    for i, (name, mpath) in enumerate(manual_specs):
        if not mpath.exists():
            raise SystemExit(f"[STOP] Manual mask does not exist: {mpath} (Name={name})")

        lv = manual_label_start + i
        _mimg, mdata = load_nifti_3d(mpath)

        if mdata.shape[:3] != ref_shape:
            raise SystemExit(
                f"[STOP] Shape mismatch: manual {mdata.shape[:3]} != ref {ref_shape}: {mpath}"
            )

        mask = mdata > 0.5
        new_vox = mask & (labels == 0)
        labels[new_vox] = int(lv)
        mapping_rows.append((int(lv), str(name)))

    if int(np.count_nonzero(labels)) == 0:
        print(f"[WARN] ALL_ROIS multilabel would be empty. Not writing: {out_img_path}")
        return

    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(labels, ref_img.affine, ref_img.header.copy())
    out_img.set_data_dtype(np.int16)
    nib.save(out_img, str(out_img_path))

    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label_id", "name"])
        for lv, nm in mapping_rows:
            w.writerow([lv, nm])

    print(f"[INFO] Wrote ALL_ROIS multilabel: {out_img_path}")
    print(f"[INFO] Wrote mapping CSV:        {out_csv_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser("Brain signal extraction pipeline (DWI - T1 coregistration + ROI extraction)")
    default_dummy_scans = read_nonnegative_int_env("DUMMY_SCANS", 0)
    default_reuse_reference = read_bool_env("REUSE_REFERENCE", True)

    ap.add_argument("--exp-root", type=Path, required=True, help="Folder containing denoised/topup DWI NIfTIs.")
    ap.add_argument("--out-root", type=Path, required=True, help="Base output folder.")
    ap.add_argument("--subjects-dir", type=Path, required=True, help="FreeSurfer SUBJECTS_DIR root (contains sub-<subj>).")
    ap.add_argument("--grad-root", type=Path, default=None, help="Folder with .bval/.bvec or .gval/.gvec. Defaults to --exp-root if omitted.")

    ap.add_argument("--dwi-glob", default="*.nii.gz", help="Glob pattern to find DWI NIfTIs within --exp-root.")
    ap.add_argument(
        "--dwi-variant",
        default="all",
        help="Which dcm2niix conflict variant to use: 'none' for no suffix, 'a', 'b', etc., or 'all'.",
    )
    ap.add_argument("--cut-token", action="append", default=[""], help="Token(s) used to derive seq_no_ext from seq_name_full.")
    ap.add_argument("--bet-f", type=float, default=0.1, help="BET fractional intensity threshold.")
    ap.add_argument("--mean", action="store_true", help="Collapse all volumes into one mean image and write one signal row per sequence.")
    ap.add_argument(
        "--dummy-scans",
        type=int,
        default=default_dummy_scans,
        help="Discard this many initial volumes when --mean computes NII_mean.nii.gz, mean.nii.gz, and collapsed ROI tables.",
    )
    ap.add_argument(
        "--reuse-reference",
        action=argparse.BooleanOptionalAction,
        default=default_reuse_reference,
        help="Reuse existing reference images. Use --no-reuse-reference to overwrite NII_mean/mean or NII_b0/b0_mean.",
    )

    ap.add_argument("--syringe-mask-t1", type=Path, default=None, help="Syringe mask in T1 space (will be warped to DWI).")
    ap.add_argument("--syringe-mask-dwi", type=Path, default=None, help="Syringe mask already in DWI space (used directly).")
    ap.add_argument("--manual-mask-dwi", action="append", default=[], help='Manual binary mask in DWI space: "Name=mask.nii.gz"')

    ap.add_argument("--atlas-roi", action="append", default=[], help='FreeSurfer atlas ROI spec: "LABEL:Name"')
    ap.add_argument("--no-default-cc", action="store_true", help="Do not auto-add CC ROIs (251..255).")

    ap.add_argument("--export-t1-only", action="store_true", help="Only export/reuse FreeSurfer T1/T1_brain/wmparc and exit.")
    ap.add_argument("--write-fslmeants", action="store_true", help="Write fslmeants .txt files in addition to the Excel table.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands but do not execute external commands.")
    ap.add_argument("--fail-on-existing", action="store_true", help="Stop if any output to be reused already exists.")

    args = ap.parse_args()

    if args.dummy_scans < 0:
        raise SystemExit(f"[STOP] --dummy-scans must be >= 0. Got {args.dummy_scans}.")
    if args.dummy_scans > 0 and not args.mean:
        raise SystemExit("[STOP] --dummy-scans can only be used together with --mean.")

    if args.syringe_mask_t1 is not None and args.syringe_mask_dwi is not None:
        raise SystemExit("[STOP] Use only one of --syringe-mask-t1 or --syringe-mask-dwi, not both.")

    exp_root = args.exp_root.resolve()
    grad_root = args.grad_root.resolve() if args.grad_root is not None else exp_root
    subjects_dir = args.subjects_dir.resolve()
    out_root = args.out_root.resolve()

    subj = exp_root.name
    fs_subj = subjects_dir / f"sub-{subj}"
    out_subj = out_root / subj
    out_subj.mkdir(parents=True, exist_ok=True)

    t1_full, t1_brain, wmparc_t1 = prep_struct_once(
        fs_subj,
        out_subj,
        dry_run=args.dry_run,
        fail_on_existing=args.fail_on_existing,
    )

    if args.export_t1_only:
        print(f"[OK] Structural export done for {subj}")
        print(f"     T1 full:   {t1_full}")
        print(f"     T1 brain:  {t1_brain}")
        print(f"     wmparc:    {wmparc_t1}")
        return

    dwis = find_dwis(exp_root, grad_root, args.dwi_glob, args.cut_token, args.dwi_variant)

    syringe_mask_t1 = None
    if args.syringe_mask_t1 is not None:
        syringe_mask_t1 = resolve_existing_path(args.syringe_mask_t1, out_subj, exp_root)
        if not syringe_mask_t1.exists():
            print(f"[WARN] syringe-mask-t1 does not exist: {syringe_mask_t1}. Ignoring.")
            syringe_mask_t1 = None

    manual_specs: list[tuple[str, Path]] = []
    for spec in args.manual_mask_dwi:
        if "=" not in spec:
            raise SystemExit(f"[STOP] Invalid --manual-mask-dwi '{spec}'. Use Name=mask.nii.gz")
        name, rel = spec.split("=", 1)
        mpath = Path(rel.strip())
        if not mpath.is_absolute():
            mpath = exp_root / mpath
        manual_specs.append((name.strip(), mpath))

    default_atlas_rois: list[tuple[int, str]] = []
    if not args.no_default_cc:
        default_atlas_rois = [
            (251, "PostCC"),
            (252, "MidPostCC"),
            (253, "CentralCC"),
            (254, "MidAntCC"),
            (255, "AntCC"),
        ]

    atlas_specs_raw = default_atlas_rois + [parse_atlas_roi_spec(spec) for spec in args.atlas_roi]
    atlas_specs = dedupe_label_specs(atlas_specs_raw)

    atlas_t1_selected: Path | None = None
    if atlas_specs:
        atlas_t1_selected = out_subj / "atlas_requested_rois_T1.nii.gz"

        if not atlas_t1_selected.exists():
            if args.dry_run:
                print(f"[DRY] Would create selected atlas in T1: {atlas_t1_selected}")
            else:
                nvox_t1 = write_selected_label_image(wmparc_t1, atlas_specs, atlas_t1_selected)
                if nvox_t1 == 0:
                    print(f"[WARN] None of the requested labels were found in wmparc (T1): {atlas_t1_selected}")
                else:
                    print(f"[INFO] Wrote selected atlas in T1: {atlas_t1_selected} (nonzero voxels: {nvox_t1})")
        else:
            print(f"[INFO] Reusing selected atlas in T1: {atlas_t1_selected}")

    for dwi_nii, grad_values, grad_vectors, grad_kind, seq_name_full, seq_no_ext in dwis:
        out_seq = out_subj / seq_no_ext
        out_seq.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Working on sequence ===\nSEQ_NAME={seq_name_full}\nSEQ_no_ext={seq_no_ext}\n")

        if args.mean:
            ref_img = make_full_mean_with_fsl(
                dwi_nii,
                out_seq,
                dummy_scans=args.dummy_scans,
                reuse_reference=args.reuse_reference,
                dry_run=args.dry_run,
                fail_on_existing=args.fail_on_existing,
            )
        elif grad_kind == "b":
            ref_img = make_b0_mean_with_mrtrix_fsl(
                dwi_nii,
                grad_vectors,
                grad_values,
                out_seq,
                reuse_reference=args.reuse_reference,
                dry_run=args.dry_run,
                fail_on_existing=args.fail_on_existing,
            )
        else:
            ref_img = make_zero_gradient_mean_from_values(
                dwi_nii,
                grad_values,
                out_seq,
                reuse_reference=args.reuse_reference,
                dry_run=args.dry_run,
                fail_on_existing=args.fail_on_existing,
            )

        b0_brain = bet_b0(
            ref_img, out_seq, seq_name_full,
            bet_f=args.bet_f,
            dry_run=args.dry_run,
            fail_on_existing=args.fail_on_existing,
        )

        affine = ants_register_b0_to_t1(
            t1_brain, b0_brain, out_seq / "b0_2_T1",
            dry_run=args.dry_run,
            fail_on_existing=args.fail_on_existing,
        )

        if not args.dry_run:
            t1_in_dwi = out_seq / f"T1_2_{seq_name_full}.nii.gz"
            t1_brain_in_dwi = out_seq / f"T1_brain_2_{seq_name_full}.nii.gz"

            ants_apply_inverse_image(
                moving_img_t1=t1_full,
                ref_dwi=ref_img,
                affine_b0_to_t1=affine,
                out_img_dwi=t1_in_dwi,
                interp="Linear",
                dry_run=args.dry_run,
                fail_on_existing=args.fail_on_existing,
            )

            ants_apply_inverse_image(
                moving_img_t1=t1_brain,
                ref_dwi=ref_img,
                affine_b0_to_t1=affine,
                out_img_dwi=t1_brain_in_dwi,
                interp="Linear",
                dry_run=args.dry_run,
                fail_on_existing=args.fail_on_existing,
            )
        else:
            print(f"[DRY] Would warp T1 and T1_brain into DWI space in: {out_seq}")

        atlas_in_dwi: Path | None = None
        if atlas_t1_selected is not None:
            atlas_in_dwi = out_seq / f"atlas_requested_rois_2_{seq_name_full}.nii.gz"
            ants_apply_inverse_label(
                moving_label_t1=atlas_t1_selected,
                ref_dwi=ref_img,
                affine_b0_to_t1=affine,
                out_label_dwi=atlas_in_dwi,
                dry_run=args.dry_run,
                fail_on_existing=args.fail_on_existing,
            )

        syr_in_dwi: Path | None = None
        if args.syringe_mask_dwi is not None:
            syr_in_dwi = resolve_existing_path(args.syringe_mask_dwi, out_seq, out_subj, exp_root)
            if not syr_in_dwi.exists():
                raise SystemExit(f"[STOP] --syringe-mask-dwi does not exist: {syr_in_dwi}")
            print(f"[INFO] Using syringe mask already in DWI space: {syr_in_dwi}")

        elif syringe_mask_t1 is not None:
            syr_in_dwi = out_seq / f"syringe_mask_2_{seq_name_full}.nii.gz"
            ants_apply_inverse_label(
                moving_label_t1=syringe_mask_t1,
                ref_dwi=ref_img,
                affine_b0_to_t1=affine,
                out_label_dwi=syr_in_dwi,
                dry_run=args.dry_run,
                fail_on_existing=args.fail_on_existing,
            )

        dwi_img = nib.load(str(dwi_nii))

        atlas_roi_masks: list[tuple[str, Path]] = []
        if atlas_in_dwi is not None and atlas_specs:
            for label_value, roi_name in atlas_specs:
                safe_name = sanitize_name(roi_name)
                mask_path = out_seq / f"{safe_name}_label{label_value}_2_{seq_name_full}.nii.gz"

                if mask_path.exists():
                    if args.fail_on_existing:
                        raise SystemExit(f"[STOP] Found existing ROI mask and --fail-on-existing was set: {mask_path}")
                    print(f"[INFO] Reusing ROI mask: {mask_path}")
                    nvox = count_nonzero_mask_voxels(mask_path)
                else:
                    if args.dry_run:
                        print(f"[DRY] Would create binary ROI mask: {mask_path}")
                        nvox = 1
                    else:
                        nvox = write_binary_mask_from_label_image(atlas_in_dwi, label_value, mask_path)

                if nvox == 0:
                    print(f"[WARN] Empty ROI after warp: {roi_name} (label={label_value}) for {seq_name_full}. Skipping.")
                    continue

                atlas_roi_masks.append((roi_name, mask_path))

        if not args.dry_run:
            all_rois_multilabel = out_seq / f"ALL_ROIS_labels_2_{seq_name_full}.nii.gz"
            all_rois_mapping = out_seq / f"ALL_ROIS_labels_2_{seq_name_full}_mapping.csv"
            build_all_rois_multilabel(
                ref_img_path=ref_img,
                atlas_label_img_path=atlas_in_dwi,
                atlas_specs=atlas_specs,
                syringe_mask_dwi=syr_in_dwi,
                manual_specs=manual_specs,
                out_img_path=all_rois_multilabel,
                out_csv_path=all_rois_mapping,
                fail_on_existing=args.fail_on_existing,
            )
        else:
            print(f"[DRY] Would create ALL_ROIS multilabel and mapping in: {out_seq}")

        if args.write_fslmeants:
            for roi_name, mask_path in atlas_roi_masks:
                out_tc = out_seq / f"{seq_name_full}_{sanitize_name(roi_name)}_tc.txt"
                fslmeants_mask(dwi_nii, mask_path, out_tc, dry_run=args.dry_run)

            if syr_in_dwi is not None:
                out_syr_tc = out_seq / f"{seq_name_full}_syringe_tc.txt"
                fslmeants_mask(dwi_nii, syr_in_dwi, out_syr_tc, dry_run=args.dry_run)

            for name, mpath in manual_specs:
                out_m_tc = out_seq / f"{seq_name_full}_{sanitize_name(name)}_tc.txt"
                fslmeants_mask(dwi_nii, mpath, out_m_tc, dry_run=args.dry_run)

        rois = []
        for roi_name, mask_path in atlas_roi_masks:
            rois.append(build_roi_from_binary_mask(mask_path, dwi_img, roi_name))

        if syr_in_dwi is not None:
            rois.append(build_roi_from_binary_mask(syr_in_dwi, dwi_img, "Syringe"))

        for name, mpath in manual_specs:
            rois.append(build_roi_from_binary_mask(mpath, dwi_img, name))

        print("[INFO] Extracting ROI tables (this can take a while)...")
        tables = extract_tables(
            dwi_nii,
            grad_values,
            rois,
            collapse_mean=args.mean,
            dummy_scans=args.dummy_scans,
        )
        print("[INFO] Writing Excel...")

        out_xlsx = out_root / "Results" / subj / f"{seq_no_ext}_results.xlsx"
        write_excel_like_matlab(tables, out_xlsx)

        print(f"[OK] Wrote: {out_xlsx}")
        print(f"[OK] {subj} / {seq_name_full} -> {out_seq}")


if __name__ == "__main__":
    main()
