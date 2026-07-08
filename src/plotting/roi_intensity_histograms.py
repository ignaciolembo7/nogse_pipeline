from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from plotting.core import distinct_colors, ensure_dir


def render_roi_intensity_histograms(
    intensity_img_path: Path,
    rois: Sequence[object],  # duck-typed: objects with .name (str) and .mask (bool ndarray)
    out_png: Path,
    *,
    title: str,
    bins: int = 50,
    ncols: int = 4,
    dpi: int = 200,
) -> Path:
    """One intensity histogram subplot per ROI, titled with its voxel count."""
    data = np.squeeze(np.asarray(nib.load(str(intensity_img_path)).get_fdata(dtype=np.float32)))

    colors = distinct_colors(len(rois))
    ncols = min(ncols, len(rois))
    nrows = -(-len(rois) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), squeeze=False)

    for i, roi in enumerate(rois):
        ax = axes[i // ncols][i % ncols]
        values = data[roi.mask]
        ax.hist(values, bins=bins, color=colors[i])
        mean_val = float(np.mean(values)) if values.size else float("nan")
        ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.2, label=f"mean={mean_val:.1f}")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"{roi.name} (n={values.size})", fontsize=9)

    for j in range(len(rois), nrows * ncols):
        axes[j // ncols][j % ncols].set_axis_off()

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png
