# Signal Rotation And Contrast Tables

This stage turns prepared signal tables into derived representations used by
the OGSE and NOGSE fitting workflows.

## Tensor-Based Rotation

The brain OGSE workflow rotates six-direction signal tables into tensor-informed
axes.

Entry point:

- `scripts/data/rotate_ogse_tensor.py`

Reusable implementation:

- `src/signal_rotation/rotation_tensor.py`
- `src/signal_rotation/dirs.py`

The rotation module groups signals by ROI and measurement step, estimates a
tensor-like directional representation, and writes rotated signal rows. Common
output directions include `long`, `tra`, tensor eigenvector directions, and
laboratory axes when requested by the caller.

The same stage can write projected diffusivity tables (`Dproj`) for later
inspection and plotting.

## Contrast Construction

Entry point:

- `scripts/data/make_contrast.py`

Reusable implementation:

- `src/fitting/contrast.py`
- `src/ogse_fitting/contrast.py`

`src/fitting/contrast.py` is the generic table-level contrast builder. It merges
two already-aligned signal tables using key columns such as:

- `stat`
- `roi`
- `direction`
- `b_step`

It writes side-specific columns with `_1` and `_2` suffixes and derives:

```text
value = value_1 - value_2
value_norm = value_norm_1 - value_norm_2
```

If normalized columns are unavailable, the function falls back to `S0_1` and
`S0_2` normalization.

## Why The Side Columns Matter

Contrast fitters need both sides of a contrast. The output table therefore keeps
side-specific metadata such as:

- `N_1`, `N_2`
- `Hz_1`, `Hz_2`
- `delta_ms_1`, `delta_ms_2`
- `Delta_app_ms_1`, `Delta_app_ms_2`
- `source_file_1`, `source_file_2`
- `sheet_1`, `sheet_2`

This is also what allows side-specific gradient correction later.

