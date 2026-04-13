from __future__ import annotations

import argparse
import re
from pathlib import Path

import nibabel as nib
import numpy as np


G_TOKEN_RE = re.compile(r"(?:^|_)G(?P<g>\d+(?:p\d+)?)(?:_|$)", re.IGNORECASE)


def parse_g_from_name(name: str) -> float:
    """Extract the gradient amplitude encoded as GXX or GXXpY from a filename."""
    match = G_TOKEN_RE.search(name)
    if match is None:
        raise ValueError(f"Could not find a G token in filename: {name}")
    return float(match.group("g").replace("p", "."))


def infer_nvol(nii_path: Path) -> int:
    """Return the number of volumes in a NIfTI file."""
    img = nib.load(str(nii_path))
    shape = tuple(int(v) for v in img.shape)
    if len(shape) == 4:
        return int(shape[3])
    if len(shape) == 3:
        return 1
    raise ValueError(f"Unsupported NIfTI shape {shape} for file: {nii_path}")


def normalize_direction(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    """Normalize a nonzero 3D direction vector."""
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError(f"Direction vector must be nonzero. Got: {vec}")
    arr = arr / norm
    return float(arr[0]), float(arr[1]), float(arr[2])


def format_row(values: list[float]) -> str:
    """Format a sequence of floats as a FSL-like whitespace-separated row."""
    return " ".join(f"{v:g}" for v in values)


def write_text(path: Path, text: str, *, overwrite: bool) -> None:
    """Write text to disk, optionally refusing to overwrite existing files."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.write_text(text + "\n", encoding="utf-8")


def parse_glob_args(raw_globs: list[str] | None) -> list[str]:
    """Expand repeated and comma-separated glob arguments into a clean list."""
    if not raw_globs:
        return ["*NOGSE*.nii.gz"]

    out: list[str] = []
    for raw in raw_globs:
        for item in str(raw).split(","):
            pat = item.strip()
            if pat:
                out.append(pat)

    return out or ["*NOGSE*.nii.gz"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate .gval and .gvec sidecars from NIfTI filenames that encode G values."
    )
    ap.add_argument("exp_root", type=Path, help="Folder containing the NIfTI sequences.")
    ap.add_argument(
        "--glob",
        action="append",
        default=None,
        help="Glob used to find target NIfTI files. Repeat the flag or use commas to pass multiple patterns.",
    )
    ap.add_argument(
        "--direction",
        nargs=3,
        type=float,
        metavar=("GX", "GY", "GZ"),
        default=(1.0, 0.0, 0.0),
        help="Common gradient direction to store in every .gvec file.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .gval/.gvec files.")
    ap.add_argument("--dry-run", action="store_true", help="Print planned outputs without writing files.")
    args = ap.parse_args()

    exp_root = args.exp_root.resolve()
    if not exp_root.is_dir():
        raise SystemExit(f"Input directory does not exist: {exp_root}")

    direction = normalize_direction(tuple(float(v) for v in args.direction))
    glob_patterns = parse_glob_args(args.glob)
    seen_paths: set[Path] = set()
    nii_paths: list[Path] = []
    for pattern in glob_patterns:
        for path in sorted(exp_root.glob(pattern)):
            if path not in seen_paths:
                seen_paths.add(path)
                nii_paths.append(path)
    if not nii_paths:
        raise SystemExit(f"No files matched glob(s) {glob_patterns!r} in {exp_root}")

    print(f"[INFO] Using glob pattern(s): {', '.join(glob_patterns)}")

    written = 0
    skipped = 0

    for nii_path in nii_paths:
        seq_name = nii_path.name
        try:
            g_value = parse_g_from_name(seq_name)
        except ValueError:
            skipped += 1
            print(f"[WARN] Skipping file without G token: {nii_path}")
            continue

        nvol = infer_nvol(nii_path)
        base = nii_path.parent / nii_path.name.replace(".nii.gz", "")
        if base.name == nii_path.name:
            base = nii_path.with_suffix("")

        gval_path = base.with_suffix(".gval")
        gvec_path = base.with_suffix(".gvec")

        gval_row = format_row([g_value] * nvol)
        gvec_rows = "\n".join(
            format_row([component] * nvol) for component in direction
        )

        print(f"[INFO] Sequence : {nii_path.name}")
        print(f"[INFO]   G value : {g_value:g}")
        print(f"[INFO]   Nvol    : {nvol}")
        print(f"[INFO]   gval    : {gval_path}")
        print(f"[INFO]   gvec    : {gvec_path}")

        if args.dry_run:
            continue

        write_text(gval_path, gval_row, overwrite=args.overwrite)
        write_text(gvec_path, gvec_rows, overwrite=args.overwrite)
        written += 1

    print()
    print("Finished.")
    print(f"  Matched files : {len(nii_paths)}")
    print(f"  Written pairs : {written}")
    print(f"  Skipped files : {skipped}")


if __name__ == "__main__":
    main()
