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
coreg_extract_phantom.py

Phantom-oriented pipeline:
1) Find DWI NIfTIs and corresponding .bval/.bvec
2) Compute b0_mean from the DWI
3) Go directly to each per-sequence output folder in Data_signals/<subject>/<sequence>/
4) Discover binary ROI masks already drawn manually in DWI space
5) Build a single multi-label image with all ROIs
6) Optionally write fslmeants timecourses
7) Extract ROI tables and write Excel

No T1, no FreeSurfer, no BET, no ANTs, no atlas warping.
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
      - 2D: (X,Y) -> promoted to (X,Y,1) for single-slice phantom data
      - 3D: (X,Y,Z)
      - 4D singleton: (X,Y,Z,1) -> squeezed to (X,Y,Z)

    Raises SystemExit for any other shape.
    """
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)

    if data.ndim == 2:
        data = data[..., np.newaxis]
        img = nib.Nifti1Image(data, img.affine, img.header.copy())

    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]

    if data.ndim != 3:
        raise SystemExit(f"[STOP] Expected 3D (or 4D with T=1) but got shape={data.shape}: {path}")

    return img, data


def is_binaryish_mask(path: Path, *, atol: float = 1e-6) -> bool:
    """
    Return True if the image is effectively binary (values close to 0 or 1).
    """
    try:
        _img, data = load_nifti_3d(path)
    except Exception:
        return False

    data = np.asanyarray(data, dtype=np.float64)
    if data.size == 0:
        return False
    if not np.all(np.isfinite(data)):
        return False

    ok = np.isclose(data, 0.0, atol=atol) | np.isclose(data, 1.0, atol=atol)
    return bool(np.all(ok))


def infer_roi_name_from_mask_path(mask_path: Path) -> str:
    """
    Infer ROI name from filename.
    """
    name = strip_nii_ext(mask_path)
    name = re.sub(r"(?i)^mask[_\-]*", "", name)
    name = re.sub(r"(?i)[_\-]*mask$", "", name)
    return name or strip_nii_ext(mask_path)


def count_nonzero_mask_voxels(mask_path: Path) -> int:
    """Count nonzero voxels in a binary mask."""
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

    Returns:
      (dwi_nii, grad_values, grad_vectors, grad_kind, seq_name_full, seq_no_ext)
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
# b0
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


# ---------------------------------------------------------------------
# Mask discovery / exports
# ---------------------------------------------------------------------
def fslmeants_mask(dwi: Path, mask: Path, out_txt: Path, dry_run: bool = False) -> None:
    """Write fslmeants timecourse for a mask."""
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    run(["fslmeants", "-i", str(dwi), "-o", str(out_txt), "-m", str(mask)], dry_run=dry_run)


def discover_sequence_masks(
    seq_dir: Path,
    dwi_nii: Path,
    mask_glob: str,
) -> list[tuple[str, Path]]:
    """
    Discover manually drawn binary masks already present inside seq_dir.

    Rules:
      - search seq_dir with mask_glob
      - ignore known pipeline products
      - ignore the DWI itself
      - keep only binary-ish NIfTI masks
      - infer ROI name from filename
    """
    reserved_substrings = (
        "NII_b0",
        "NII_mean",
        "b0_mean",
        "mean.nii.gz",
        "_b0_brain",
        "b0_2_T1",
        "T1_2_",
        "T1_brain_2_",
        "atlas_requested_rois",
        "ALL_ROIS_labels",
        "wmparc",
        "_results",
    )

    out: list[tuple[str, Path]] = []
    seen_names: set[str] = set()
    dwi_abs = dwi_nii.resolve()

    for p in sorted(seq_dir.glob(mask_glob)):
        if not p.is_file():
            continue

        if p.resolve() == dwi_abs:
            continue

        if any(tok in p.name for tok in reserved_substrings):
            continue

        if not is_binaryish_mask(p):
            continue

        roi_name = infer_roi_name_from_mask_path(p)
        if roi_name in seen_names:
            raise SystemExit(
                f"[STOP] Duplicate ROI name discovered in phantom mode: '{roi_name}' in {seq_dir}. "
                "Rename the masks so each ROI has a unique filename."
            )

        seen_names.add(roi_name)
        out.append((roi_name, p))

    return out


def build_multilabel_from_binary_masks(
    ref_img_path: Path,
    mask_specs: list[tuple[str, Path]],
    out_img_path: Path,
    out_csv_path: Path,
    *,
    fail_on_existing: bool,
    label_start: int = 9000,
) -> None:
    """
    Build a single 3D multi-label NIfTI in DWI space from binary masks:
      mask_specs = [(roi_name, mask_path), ...]

    Labels are assigned as 9000, 9001, ...
    Also writes a CSV mapping {label_id -> name}.
    """
    nii_exists = out_img_path.exists()
    csv_exists = out_csv_path.exists()

    if nii_exists and fail_on_existing:
        raise SystemExit(f"[STOP] Found existing {out_img_path} and --fail-on-existing was set.")
    if csv_exists and fail_on_existing:
        raise SystemExit(f"[STOP] Found existing {out_csv_path} and --fail-on-existing was set.")

    # In phantom mode we always regenerate ALL_ROIS so the multilabel stays
    # synchronized with the binary masks currently present in the sequence folder.
    if nii_exists and csv_exists:
        print(f"[INFO] Rebuilding existing ALL_ROIS multilabel: {out_img_path}")
        print(f"[INFO] Rebuilding existing mapping CSV:        {out_csv_path}")
        out_img_path.unlink()
        out_csv_path.unlink()
    elif nii_exists and not csv_exists:
        print("[INFO] ALL_ROIS NIfTI exists but mapping CSV is missing; rebuilding both for consistency.")
        out_img_path.unlink()
    elif csv_exists and not nii_exists:
        print("[INFO] ALL_ROIS mapping CSV exists but NIfTI is missing; rebuilding both for consistency.")
        out_csv_path.unlink()

    ref_img, _ = load_nifti_3d(ref_img_path)
    ref_shape = ref_img.shape[:3]
    labels = np.zeros(ref_shape, dtype=np.int16)

    mapping_rows: list[tuple[int, str]] = []

    for i, (roi_name, mask_path) in enumerate(mask_specs):
        _mimg, mdata = load_nifti_3d(mask_path)

        if mdata.shape[:3] != ref_shape:
            raise SystemExit(
                f"[STOP] Shape mismatch: mask {mdata.shape[:3]} != ref {ref_shape}: {mask_path}"
            )

        label_value = label_start + i
        mask = mdata > 0.5
        new_vox = mask & (labels == 0)
        labels[new_vox] = int(label_value)
        mapping_rows.append((int(label_value), str(roi_name)))

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
    ap = argparse.ArgumentParser("Phantom signal extraction pipeline (direct ROI extraction in DWI space)")
    default_dummy_scans = read_nonnegative_int_env("DUMMY_SCANS", 0)
    default_reuse_reference = read_bool_env("REUSE_REFERENCE", True)

    ap.add_argument("--exp-root", type=Path, required=True, help="Folder containing DWI NIfTIs.")
    ap.add_argument("--out-root", type=Path, required=True, help="Base output folder.")
    ap.add_argument("--out-subj", type=Path, default=None, help="Relative subject folder under --out-root. Defaults to the --exp-root folder name.")
    ap.add_argument("--grad-root", type=Path, default=None, help="Folder with .bval/.bvec or .gval/.gvec. Defaults to --exp-root if omitted.")

    ap.add_argument("--dwi-glob", default="*.nii.gz", help="Glob pattern to find DWI NIfTIs within --exp-root.")
    ap.add_argument(
        "--dwi-variant",
        default="all",
        help="Which dcm2niix conflict variant to use: 'none' for no suffix, 'a', 'b', etc., or 'all'.",
    )
    ap.add_argument("--cut-token", action="append", default=[""], help="Token(s) used to derive seq_no_ext from seq_name_full.")
    ap.add_argument("--mask-glob", default="*mask*.nii*", help="Glob used inside each per-sequence output folder to discover manually drawn binary masks.")
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

    ap.add_argument("--write-fslmeants", action="store_true", help="Write fslmeants .txt files in addition to the Excel table.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands but do not execute external commands.")
    ap.add_argument("--fail-on-existing", action="store_true", help="Stop if any output to be reused already exists.")

    args = ap.parse_args()

    if args.dummy_scans < 0:
        raise SystemExit(f"[STOP] --dummy-scans must be >= 0. Got {args.dummy_scans}.")
    if args.dummy_scans > 0 and not args.mean:
        raise SystemExit("[STOP] --dummy-scans can only be used together with --mean.")

    exp_root = args.exp_root.resolve()
    grad_root = args.grad_root.resolve() if args.grad_root is not None else exp_root
    out_root = args.out_root.resolve()

    if args.out_subj is not None:
        if args.out_subj.is_absolute() or ".." in args.out_subj.parts:
            raise SystemExit("[STOP] --out-subj must be a relative path inside --out-root.")
        subj = str(args.out_subj)
    else:
        subj = exp_root.name
    out_subj = out_root / subj
    out_subj.mkdir(parents=True, exist_ok=True)

    dwis = find_dwis(exp_root, grad_root, args.dwi_glob, args.cut_token, args.dwi_variant)

    for dwi_nii, grad_values, grad_vectors, grad_kind, seq_name_full, seq_no_ext in dwis:
        out_seq = out_subj / seq_no_ext
        out_seq.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Working on phantom sequence ===\nSEQ_NAME={seq_name_full}\nSEQ_no_ext={seq_no_ext}\n")

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

        mask_specs = discover_sequence_masks(out_seq, dwi_nii, args.mask_glob)
        if not mask_specs:
            raise SystemExit(
                f"[STOP] No binary masks were found inside:\n"
                f"  {out_seq}\n"
                f"using glob:\n"
                f"  {args.mask_glob}\n\n"
                "Draw the phantom masks in DWI space and place them in that per-sequence folder."
            )

        print(f"[INFO] Found {len(mask_specs)} phantom mask(s) in {out_seq}")
        for roi_name, mask_path in mask_specs:
            nvox = count_nonzero_mask_voxels(mask_path)
            print(f"[INFO]   ROI: {roi_name} -> {mask_path} (nonzero voxels: {nvox})")

        if not args.dry_run:
            all_rois_multilabel = out_seq / f"ALL_ROIS_labels_2_{seq_name_full}.nii.gz"
            all_rois_mapping = out_seq / f"ALL_ROIS_labels_2_{seq_name_full}_mapping.csv"
            build_multilabel_from_binary_masks(
                ref_img_path=ref_img,
                mask_specs=mask_specs,
                out_img_path=all_rois_multilabel,
                out_csv_path=all_rois_mapping,
                fail_on_existing=args.fail_on_existing,
            )
        else:
            print(f"[DRY] Would create ALL_ROIS multilabel and mapping in: {out_seq}")

        if args.write_fslmeants:
            for roi_name, mask_path in mask_specs:
                out_tc = out_seq / f"{seq_name_full}_{sanitize_name(roi_name)}_tc.txt"
                fslmeants_mask(dwi_nii, mask_path, out_tc, dry_run=args.dry_run)

        if args.dry_run:
            print(f"[DRY] Would extract ROI tables from {len(mask_specs)} masks and write Excel in {out_seq}")
            continue

        dwi_img = nib.load(str(dwi_nii))
        rois = [build_roi_from_binary_mask(mask_path, dwi_img, roi_name) for roi_name, mask_path in mask_specs]

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
        print(f"[OK] Phantom {subj} / {seq_name_full} -> {out_seq}")


if __name__ == "__main__":
    main()
