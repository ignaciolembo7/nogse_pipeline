"""Metadata readers and generated sidecar writers for DWI preprocessing."""

from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path
from typing import Iterable


PHASE_ENCODING_TO_VECTOR = {
    "i": (1, 0, 0),
    "i-": (-1, 0, 0),
    "j": (0, 1, 0),
    "j-": (0, -1, 0),
    "k": (0, 0, 1),
    "k-": (0, 0, -1),
}


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON metadata file: {path}")
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def phase_encoding_vector(metadata: dict) -> tuple[int, int, int, float]:
    try:
        direction = metadata["PhaseEncodingDirection"]
    except KeyError as exc:
        raise KeyError("JSON metadata is missing PhaseEncodingDirection") from exc

    try:
        readout_time = float(metadata["TotalReadoutTime"])
    except KeyError as exc:
        raise KeyError("JSON metadata is missing TotalReadoutTime") from exc

    if direction not in PHASE_ENCODING_TO_VECTOR:
        valid = ", ".join(sorted(PHASE_ENCODING_TO_VECTOR))
        raise ValueError(f"Unsupported PhaseEncodingDirection '{direction}'. Valid values: {valid}")

    vector = PHASE_ENCODING_TO_VECTOR[direction]
    return (*vector, readout_time)


def write_acqparams(json_paths: Iterable[Path], output_path: Path) -> None:
    rows = [phase_encoding_vector(read_json(path)) for path in json_paths]
    if not rows:
        raise ValueError("At least one JSON metadata file is required to write acqparams")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(f"{row[0]} {row[1]} {row[2]} {row[3]:.6f}\n")


def read_nifti_volume_count(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Missing NIfTI image: {path}")

    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as stream:
        header = stream.read(352)

    if len(header) < 56:
        raise ValueError(f"File is too small to be a valid NIfTI image: {path}")

    sizeof_hdr_le = struct.unpack("<i", header[:4])[0]
    sizeof_hdr_be = struct.unpack(">i", header[:4])[0]
    if sizeof_hdr_le == 348:
        endian = "<"
    elif sizeof_hdr_be == 348:
        endian = ">"
    else:
        raise ValueError(f"Unsupported or invalid NIfTI header in: {path}")

    dims = struct.unpack(f"{endian}8h", header[40:56])
    ndim = dims[0]
    if ndim < 3:
        raise ValueError(f"NIfTI image has fewer than 3 dimensions: {path}")

    if ndim < 4 or dims[4] <= 0:
        return 1
    return int(dims[4])


def read_bval_count(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"Missing bval file: {path}")
    values = path.read_text(encoding="utf-8").split()
    if not values:
        raise ValueError(f"Empty bval file: {path}")
    return len(values)


def write_eddy_index(output_path: Path, n_volumes: int, *, acqparam_row: int = 1) -> None:
    if n_volumes <= 0:
        raise ValueError(f"n_volumes must be positive, got {n_volumes}")
    if acqparam_row <= 0:
        raise ValueError(f"acqparam_row must be positive, got {acqparam_row}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(" ".join([str(acqparam_row)] * n_volumes) + "\n", encoding="utf-8")


def validate_bvals_match_image(dwi_path: Path, bval_path: Path) -> int:
    n_volumes = read_nifti_volume_count(dwi_path)
    n_bvals = read_bval_count(bval_path)
    if n_volumes != n_bvals:
        raise ValueError(
            f"DWI volume count ({n_volumes}) does not match bval count ({n_bvals}): "
            f"{dwi_path} vs {bval_path}"
        )
    return n_volumes
