### Converting gradient amplitude into `b`-value

Many later steps work on a `b`-value axis even when the acquisition was organized by gradient amplitude. The repository uses:

```python
b = N * gamma**2 * delta_ms**2 * delta_app_ms * g**2 / 1e9
```

This is implemented in `src/fitting/b_from_g.py`.

Conceptually:

- `g` sets gradient strength,
- `delta` and `Delta_app` set the timing scale of the encoding,
- `N` accounts for the number of oscillation periods or lobes,
- the resulting `b` gives the diffusion-weighting strength.

The repository keeps several related axes:

- `g`
- `g_max`
- `g_lin_max`
- `g_thorsten`
- `bvalue`
- `bvalue_g`
- `bvalue_g_lin_max`
- `bvalue_thorsten`

This lets the same data be re-expressed in the axis most appropriate for plotting or fitting.

### Contrast construction

The generic contrast definition is always a difference between matched signals:

```python
value = value_1 - value_2
value_norm = value_norm_1 - value_norm_2
```

implemented in `src/fitting/contrast.py` and used by `scripts/make_contrast.py`.

The important scientific point is that the pipeline treats contrast as a derived observable built from two already-aligned experiments. The two sides are kept explicitly in the table, so later fits still know:

- which side was acquisition 1 and which was acquisition 2,
- each side's `N`, `Hz`, `sequence`, `source_file`, and gradient axis.

The core implementation is intentionally minimal:

```python
merged["value"] = merged[f"{y_col}_1"] - merged[f"{y_col}_2"]

if have_norm:
    merged["value_norm"] = merged[f"{y_norm_col}_1"] - merged[f"{y_norm_col}_2"]
else:
    merged["value_norm"] = (merged[f"{y_col}_1"] / merged["S0_1"]) - (merged[f"{y_col}_2"] / merged["S0_2"])
```

Code reference: `src/fitting/contrast.py`, `make_contrast`.