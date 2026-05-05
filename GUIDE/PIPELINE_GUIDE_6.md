

### Stage 7: OGSE contrast construction

**What goes in**

- matched rotated OGSE signal tables

**What happens conceptually**

- manually selected pairs of acquisitions are subtracted to build an OGSE contrast curve;
- the paired files share the same subject and diffusion-time context but differ in oscillation setting, typically through `N`, `Hz`, or both;
- both sides of the subtraction remain attached to the output table.

**What comes out**

- long-form OGSE contrast tables under `contrast-data-rotated/tables/...`

**Why this step is needed**

- the contrast isolates how the signal changes when the oscillating encoding is changed while keeping the broader acquisition context fixed.

**Key physical or mathematical idea**

- the contrast is treated as the observable to be fitted, not as an intermediate plotting convenience.

**Code**

- `bash_template/brains/3.1-run_make_contrast_selected_rotated.sh`
- `scripts/make_contrast.py`
- `src/fitting/contrast.py`

### Stage 8: OGSE signal fitting and macro-scale summaries

**What goes in**

- rotated signal tables

**What happens conceptually**

- the OGSE signal workflow (`scripts/fit_ogse_signal_vs_g.py`) selects a fitting model explicitly with `--model`;
- in the `4.x` batch scripts, that model is `monoexp`, implemented by the true monoexponential machinery in `src/monoexp_fitting`;
- optional OGSE-specific fitting remains available as `free_ogse`, implemented in `src/ogse_fitting` and evaluated with `M_ogse_free`;
- the fit can use either a fixed number of lowest-`b` points or an automatically selected prefix of the curve;
- from the fitted attenuation, the pipeline produces:
  - `D0`
  - synthetic `D_proj` tables
  - `D` vs `Delta_app` plots
  - `alpha_macro = <D0> / D0_ref`

**What comes out**

- OGSE signal fit tables (model-tagged)
- `Dproj` tables
- `D`-vs-`Delta_app` plots
- `summary_alpha_values.xlsx`

**Why this step is needed**

- it provides a simple diffusion-scale reference before moving to more specialized contrast models;
- `alpha_macro` is later used as a fixed slope parameter in the final `t_c`-vs-`t_d` fit.

**Key physical or mathematical idea**

For the monoexponential model used in `4.x`, the signal model is:

```text
S(b) = M0 * exp(-b * D0)
```

This is evaluated by `src/monoexp_fitting/fit_monoexp_signal_vs_bval.py`
inside the OGSE signal workflow.

For the optional OGSE-specific model (`free_ogse`), the implemented model is:

```text
S(G) = M_ogse_free(TE, G, N, TE/N, M0, D0)
```

In the centralized model module this is:

```python
def M_ogse_free(TE, G, N, x, M0, D0):
    y = TE - (N - 1) * x
    return M0 * np.exp(-1.0 / 12 * g**2 * G**2 * D0 * ((N - 1) * x**3 + y**3))
```

When `free_ogse` is selected, the fitter evaluates that model at `x = TE/N`,
using the corrected gradient axis directly. Because
the exponent is still proportional to `G^2`, these fits continue to provide the
same kind of reference diffusivity scale used later in the correction and
summary stages.

The automatic prefix selection still stops when adding another point worsens
`rmse_log` beyond a tolerance, so the fit remains in the low-attenuation part
of the OGSE curve that is intended to define the reference `D0`.

**Code**

- `scripts/fit_ogse_signal_vs_g.py`
- `src/monoexp_fitting/fit_monoexp_signal_vs_bval.py`
- `src/ogse_fitting/fit_ogse_signal_vs_g.py`
- `scripts/plot_D0_vs_Delta.py`
- `scripts/make_alpha_macro_summary.py`
- `src/tc_fittings/alpha_macro_summary.py`

### Stage 9: NOGSE-style contrast construction and fitting in brains

**What goes in**

- selected rotated signal tables
- later, optionally corrected gradient factors

**What happens conceptually**

- the repository builds additional contrast tables from selected rotated acquisitions;
- these contrasts are then fitted with three model families:
  - `free`
  - `tort`
  - `rest`

For brains, the practical sequence is:

1. fit the uncorrected `free` model,
2. build a gradient-correction table from the syringe ROI,
3. refit with correction,
4. fit the corrected `rest` model over all target ROIs.

**What comes out**

- per-analysis fit tables with parameters such as `D0`, `alpha`, `tc_ms`
- peak-derived quantities such as `tc_peak_ms`, `lcf_peak_m`, and peak gradient location

**Why this step is needed**

- the contrast models are the part of the pipeline that tries to map the measured contrast to microstructural timescales rather than just to an effective diffusivity.

**Key physical or mathematical idea**

The model formulas come from `src/models/model_fitting.py`. In the repository’s naming:

- `free` uses unrestricted diffusion with effective diffusivity `D0`,
- `tort` scales the free-diffusion contribution by `alpha`,
- `rest` introduces a finite correlation time `t_c`.

Peak metrics are then converted to characteristic length and time scales using the notebook-derived formulas implemented in:

- `src/ogse_fitting/fit_ogse_contrast_vs_g.py`
- `src/nogse_fitting/fit_nogse_contrast_vs_g.py`

**Code**

- `bash_template/brains/5.0-run_make_nogse_contrast_selected_rotated.sh`
- `bash_template/brains/5.1-run_fit_nogse_contrast_vs_g.sh`
- `bash_template/brains/6.1-run_fit_rest_all_ogse_contrast_vs_g_corr.sh`
- `src/ogse_fitting/fit_ogse_contrast_vs_g.py`
- `src/nogse_fitting/fit_nogse_contrast_vs_g.py`
- `src/models/model_fitting.py`