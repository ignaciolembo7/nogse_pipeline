from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from plottings.core import compact_float, ensure_dir, sanitize_token


@dataclass(frozen=True)
class SignalImageEntry:
    sequence_dir: Path
    sequence_name: str
    image_path: Path
    gradient_value: float


@dataclass(frozen=True)
class GridCell:
    row_label: str
    gradient_value: float
    image: np.ndarray | None
    source: str
    is_difference: bool


def _normalize_text(value: object) -> str:
    text = str(value).lower()
    return re.sub(r"[^a-z0-9.]+", " ", text).strip()


def _label_tokens(label: str) -> list[str]:
    return [token for token in _normalize_text(label).split() if token]


def _gradient_regex(gradient_type: str) -> re.Pattern[str]:
    gtype = str(gradient_type).strip().lower()
    if gtype == "g":
        return re.compile(r"(?:^|_)G(\d+(?:p\d+|\.\d+)?)(?=_|$)", re.IGNORECASE)
    if gtype in {"b", "bvalue", "bval"}:
        return re.compile(r"(?:^|_)b(?:val(?:ue)?)?(\d+(?:p\d+|\.\d+)?)(?=_|$)", re.IGNORECASE)
    raise ValueError(f"Unsupported gradient type: {gradient_type!r}. Use 'g' or 'b'.")


def _parse_number_token(value: str) -> float:
    return float(str(value).replace("p", "."))


def _sequence_gradient(sequence_name: str, gradient_type: str) -> float | None:
    match = _gradient_regex(gradient_type).search(sequence_name)
    if not match:
        return None
    return _parse_number_token(match.group(1))


def _matches_tokens(sequence_name: str, label: str) -> bool:
    normalized = _normalize_text(sequence_name)
    return all(token in normalized.split() for token in _label_tokens(label))


def _passes_include_exclude(sequence_name: str, include_tokens: Sequence[str], exclude_tokens: Sequence[str]) -> bool:
    normalized = _normalize_text(sequence_name)
    sequence_tokens = set(normalized.split())

    def has_filter_token(token: str) -> bool:
        normalized_token = _normalize_text(token)
        token_parts = normalized_token.split()
        if len(token_parts) == 1:
            return normalized_token in sequence_tokens
        return normalized_token in normalized

    include_ok = all(has_filter_token(token) for token in include_tokens)
    exclude_ok = not any(has_filter_token(token) for token in exclude_tokens)
    return include_ok and exclude_ok


def collect_signal_image_entries(
    case_root: Path,
    *,
    gradient_type: str,
    image_name: str,
    include_tokens: Sequence[str] = (),
    exclude_tokens: Sequence[str] = (),
) -> list[SignalImageEntry]:
    entries: list[SignalImageEntry] = []
    for sequence_dir in sorted(p for p in case_root.iterdir() if p.is_dir()):
        sequence_name = sequence_dir.name
        if not _passes_include_exclude(sequence_name, include_tokens, exclude_tokens):
            continue

        gradient_value = _sequence_gradient(sequence_name, gradient_type)
        if gradient_value is None:
            continue

        image_path = sequence_dir / image_name
        if image_path.exists():
            entries.append(
                SignalImageEntry(
                    sequence_dir=sequence_dir,
                    sequence_name=sequence_name,
                    image_path=image_path,
                    gradient_value=gradient_value,
                )
            )
    return entries


def _find_entry(
    entries: Sequence[SignalImageEntry],
    *,
    row_label: str,
    gradient_value: float,
    gradient_tol: float,
) -> SignalImageEntry | None:
    matches = [
        entry
        for entry in entries
        if _matches_tokens(entry.sequence_name, row_label)
        and np.isclose(entry.gradient_value, gradient_value, rtol=0.0, atol=gradient_tol)
    ]
    if not matches:
        return None
    if len(matches) > 1:
        names = "\n  - ".join(entry.sequence_name for entry in matches)
        raise ValueError(
            f"More than one image matched row={row_label!r}, gradient={gradient_value:g}:\n"
            f"  - {names}\n"
            "Add --include-token or --exclude-token to disambiguate the acquisition."
        )
    return matches[0]


def _load_display_slice(
    image_path: Path,
    *,
    slice_axis: int,
    slice_index: int | None,
    volume_index: int | None,
) -> np.ndarray:
    data = np.asarray(nib.load(str(image_path)).get_fdata(dtype=np.float32))
    data = np.squeeze(data)

    if data.ndim == 4:
        if volume_index is None:
            data = np.nanmean(data, axis=3)
        else:
            data = data[:, :, :, int(volume_index)]

    if data.ndim == 2:
        out = data
    elif data.ndim == 3:
        axis = int(slice_axis)
        if axis not in {0, 1, 2}:
            raise ValueError(f"--slice-axis must be 0, 1, or 2. Got {slice_axis}.")
        idx = data.shape[axis] // 2 if slice_index is None else int(slice_index)
        if idx < 0 or idx >= data.shape[axis]:
            raise ValueError(f"Slice index {idx} is out of bounds for axis {axis} with size {data.shape[axis]}.")
        out = np.take(data, idx, axis=axis)
    else:
        raise ValueError(f"Unsupported image shape after squeezing {data.shape}: {image_path}")

    out = np.asarray(out, dtype=float)
    out[~np.isfinite(out)] = np.nan
    return np.rot90(out)


def _nonzero_bbox(images: Sequence[np.ndarray]) -> tuple[slice, slice] | None:
    masks = []
    for image in images:
        finite = np.isfinite(image)
        if not finite.any():
            continue
        scale = np.nanmax(np.abs(image[finite]))
        if not np.isfinite(scale) or scale == 0:
            continue
        masks.append(finite & (np.abs(image) > scale * 1e-6))
    if not masks:
        return None

    mask = np.logical_or.reduce(masks)
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return None

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad_y = max(2, int(round((y1 - y0) * 0.08)))
    pad_x = max(2, int(round((x1 - x0) * 0.08)))
    y0 = max(0, y0 - pad_y)
    y1 = min(mask.shape[0], y1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(mask.shape[1], x1 + pad_x)
    return slice(y0, y1), slice(x0, x1)


def _split_difference(row_label: str) -> tuple[str, str] | None:
    parts = re.split(r"\s+-\s+", row_label.strip(), maxsplit=1)
    if len(parts) == 2 and all(part.strip() for part in parts):
        return parts[0].strip(), parts[1].strip()
    return None


def build_grid_cells(
    entries: Sequence[SignalImageEntry],
    *,
    row_labels: Sequence[str],
    gradient_values: Sequence[float],
    gradient_tol: float,
    allow_missing: bool,
    slice_axis: int,
    slice_index: int | None,
    volume_index: int | None,
) -> list[list[GridCell]]:
    rows: list[list[GridCell]] = []
    image_cache: dict[Path, np.ndarray] = {}

    def load_entry(entry: SignalImageEntry) -> np.ndarray:
        if entry.image_path not in image_cache:
            image_cache[entry.image_path] = _load_display_slice(
                entry.image_path,
                slice_axis=slice_axis,
                slice_index=slice_index,
                volume_index=volume_index,
            )
        return image_cache[entry.image_path]

    for row_label in row_labels:
        row: list[GridCell] = []
        diff_parts = _split_difference(row_label)
        for gradient_value in gradient_values:
            if diff_parts is None:
                entry = _find_entry(
                    entries,
                    row_label=row_label,
                    gradient_value=float(gradient_value),
                    gradient_tol=gradient_tol,
                )
                if entry is None:
                    if not allow_missing:
                        raise ValueError(f"No image matched row={row_label!r}, gradient={gradient_value:g}.")
                    row.append(GridCell(row_label, float(gradient_value), None, "missing", False))
                else:
                    row.append(
                        GridCell(
                            row_label,
                            float(gradient_value),
                            load_entry(entry),
                            str(entry.image_path),
                            False,
                        )
                    )
            else:
                left, right = diff_parts
                left_entry = _find_entry(entries, row_label=left, gradient_value=float(gradient_value), gradient_tol=gradient_tol)
                right_entry = _find_entry(entries, row_label=right, gradient_value=float(gradient_value), gradient_tol=gradient_tol)
                if left_entry is None or right_entry is None:
                    if not allow_missing:
                        missing = left if left_entry is None else right
                        raise ValueError(f"No image matched row={missing!r}, gradient={gradient_value:g}.")
                    row.append(GridCell(row_label, float(gradient_value), None, "missing", True))
                else:
                    left_image = load_entry(left_entry)
                    right_image = load_entry(right_entry)
                    if left_image.shape != right_image.shape:
                        raise ValueError(
                            f"Cannot subtract images with different shapes for row={row_label!r}, "
                            f"gradient={gradient_value:g}: {left_image.shape} vs {right_image.shape}"
                        )
                    row.append(
                        GridCell(
                            row_label,
                            float(gradient_value),
                            left_image - right_image,
                            f"{left_entry.image_path} - {right_entry.image_path}",
                            True,
                        )
                    )
        rows.append(row)
    return rows


def _percentile_limits(images: Sequence[np.ndarray], percentile: float) -> tuple[float, float]:
    arrays = [image[np.isfinite(image)].ravel() for image in images if np.isfinite(image).any()]
    if not arrays:
        return 0.0, 1.0
    values = np.concatenate(arrays)
    if values.size == 0:
        return 0.0, 1.0
    vmax = float(np.nanpercentile(values, percentile))
    vmin = float(np.nanmin(values))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return 0.0, 1.0
    return vmin, vmax


def render_signal_image_grid(
    rows: Sequence[Sequence[GridCell]],
    *,
    gradient_type: str,
    title: str,
    out_png: Path,
    crop_nonzero: bool = True,
    intensity_percentile: float = 99.0,
    diff_percentile: float = 99.0,
    dpi: int = 220,
) -> Path:
    if not rows or not rows[0]:
        raise ValueError("The image grid is empty.")

    all_images = [cell.image for row in rows for cell in row if cell.image is not None]
    crop = _nonzero_bbox(all_images) if crop_nonzero else None
    if crop is not None:
        rows = [
            [
                GridCell(cell.row_label, cell.gradient_value, None if cell.image is None else cell.image[crop], cell.source, cell.is_difference)
                for cell in row
            ]
            for row in rows
        ]

    base_images = [cell.image for row in rows for cell in row if cell.image is not None and not cell.is_difference]
    diff_images = [cell.image for row in rows for cell in row if cell.image is not None and cell.is_difference]
    base_vmin, base_vmax = _percentile_limits(base_images, intensity_percentile)
    _diff_min, diff_vmax = _percentile_limits([np.abs(image) for image in diff_images], diff_percentile) if diff_images else (0.0, 1.0)
    diff_vmax = max(abs(float(diff_vmax)), 1e-12)

    nrows = len(rows)
    ncols = len(rows[0])
    fig_width = max(6.0, 1.35 * ncols + 1.0)
    fig_height = max(2.2, 1.35 * nrows + 0.8)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)

    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            ax = axes[r][c]
            ax.set_axis_off()
            if cell.image is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8, color="#64748b")
            elif cell.is_difference:
                ax.imshow(cell.image, cmap="gray", vmin=-diff_vmax, vmax=diff_vmax, interpolation="nearest")
            else:
                ax.imshow(cell.image, cmap="gray", vmin=base_vmin, vmax=base_vmax, interpolation="nearest")

            if r == 0:
                unit = "mT/m" if str(gradient_type).strip().lower() == "g" else "s/mm2"
                ax.set_title(f"{compact_float(cell.gradient_value)} {unit}", fontsize=9, fontweight="bold", pad=3)
            if c == 0:
                ax.text(
                    -0.12,
                    0.5,
                    cell.row_label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=9,
                )

    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout(rect=(0.03, 0.0, 1.0, 0.94), w_pad=0.25, h_pad=0.55)
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def write_grid_manifest(rows: Sequence[Sequence[GridCell]], out_csv: Path) -> Path:
    records = [
        {
            "row_label": cell.row_label,
            "gradient_value": cell.gradient_value,
            "is_difference": cell.is_difference,
            "source": cell.source,
        }
        for row in rows
        for cell in row
    ]
    ensure_dir(out_csv.parent)
    pd.DataFrame.from_records(records).to_csv(out_csv, index=False)
    return out_csv


def default_output_stem(experiment: str, name: str, gradient_type: str, row_labels: Sequence[str], gradient_values: Sequence[float]) -> str:
    values = "-".join(sanitize_token(compact_float(value)) for value in gradient_values)
    rows = "-".join(sanitize_token(label) for label in row_labels)
    return ".".join(
        [
            sanitize_token(experiment),
            sanitize_token(name),
            f"grad-{sanitize_token(gradient_type)}",
            f"values-{values}",
            f"rows-{rows}",
        ]
    )
