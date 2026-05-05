### Stage 10: Gradient correction

**What goes in**

- OGSE reference `D0` fits computed from signal curves in a reference ROI
- free-model contrast fits from the same reference ROI

**What happens conceptually**

- the repository compares a reference diffusivity inferred from OGSE signal fits with the diffusivity implied by each side of the contrast fit;
- from that comparison it computes side-specific correction factors that act on the gradient axis.

The implemented formula is:

```text
ratio_side = D0_fit_nogse_side / D0_fit_monoexp
correction_factor_side = sqrt(ratio_side)
```

The table is built explicitly as:

```python
out['ratio_1'] = out['D0_fit_nogse_1'] / out['D0_fit_monoexp']
out['ratio_2'] = out['D0_fit_nogse_2'] / out['D0_fit_monoexp']
out['correction_factor_1'] = np.sqrt(out['ratio_1'])
out['correction_factor_2'] = np.sqrt(out['ratio_2'])
```

`D0_fit_monoexp` is a single shared reference per
`subj + sheet + roi + td_ms`: it is averaged over the selected monoexp
reference `N` values (default `1,4,8`) and across available directions
(e.g., `long` and `tra`). Therefore, rows that differ only by direction use
the same `D0_fit_monoexp`.

Code reference: `src/ogse_fitting/make_grad_correction_table.py`.

In the canonical batch runners, this reference branch is wired explicitly to
the monoexponential OGSE signal outputs:
`fits/ogse_signal_vs_g_monoexp` (see `bash_template/*/5.2-run_make_grad_correction_table.sh`).

#### Why the square root appears

Here the reference `D0` fit means the Stage 8 OGSE signal fit

```text
S(b) = M0 * exp(-b * D0)
```

applied to OGSE signal curves. It is a reference `D0` estimate, not a separate
contrast-model family.

The repository uses

```text
b = C * g^2
```

with

```text
C = N * gamma^2 * delta^2 * Delta_app / 1e9
```

implemented in `src/fitting/b_from_g.py` as:

```python
return N * (gamma**2) * (delta_ms**2) * delta_app_ms * (g**2) / 1e9
```

If the effective gradient is `g_eff = f * g_nom`, then

```text
b_eff = C * g_eff^2 = f^2 * b_nom
```

so a monoexponential fit performed on the wrong `b` axis absorbs the error into
an apparent diffusivity

```text
D_app = f^2 * D_true
```

which gives

```text
f = sqrt(D_app / D_true)
```

and, in the correction table,

```text
f = sqrt(D0_fit_nogse_side / D0_fit_monoexp)
```

That square root is therefore the conversion from a diffusivity mismatch to a
gradient-amplitude correction.

#### Shared correction flow used by the four `*_vs_g` fitters

After the correction factors are computed, the four fitters now follow one
common rule:

1. choose a gradient family (`g`, `g_lin_max`, `g_thorsten`, or the
   corresponding `bvalue` representation),
2. apply correction on the gradient first,
3. derive the corrected `b` axis from that corrected gradient if a `bvalue`
   axis is requested,
4. use those corrected variables consistently for fitting and plotting.

That shared logic is centralized in `src/fitting/b_from_g.py`:

```python
gradient_raw = extract_gradient_array(df, axis=axis_base, side=resolved_side)
gradient_corr = gradient_raw * float(f_corr)

if axis_uses_bvalue(axis_base):
    bvalue_corr = bvalue_from_gradient(
        gradient_corr,
        axis=axis_base,
        N=N,
        gamma=gamma,
        delta_ms=delta_ms,
        Delta_app_ms=Delta_app_ms,
    )

axis_corr = bvalue_corr if axis_uses_bvalue(axis_base) else gradient_corr
```

The practical consequence is that correction remains tied to gradient
calibration, but each model family consumes that correction in its own natural
fit variable (`b` for monoexp signal fits, `g` for OGSE/NOGSE free-signal
fits, side-specific variables for contrast fits).