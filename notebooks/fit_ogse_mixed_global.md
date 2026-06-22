# OGSE Fitting Manifest — `M_ogse_mixed_offset`

Covers four notebooks that share the same model but differ in the scope at which
parameters are estimated.

---

## Model

**`M_ogse_mixed_offset`** — defined in `src/models/model_fitting.py`:

```
M_ogse_mixed_offset(td, G, N, x, tc, α, M0, D0, C, RN)
    = sqrt( [M0 · A_free · A_rest + C]² + RN² )
```

where `x = td / N` and:

```
A_free(td, G, N, x, D_free)
    = exp( −γ²G²D_free/12 · [(N−1)·x³ + y³] )          y = td − (N−1)·x,  D_free = α·D0

A_rest(td, G, N, x, tc, D_rest)
    = exp( −φ_SE − φ_N + φ_cross )                       D_rest = (1−α)·D0
```

The three log-attenuation phases φ_SE, φ_N, φ_cross are the Callaghan restricted-diffusion
terms computed by `_rest_log_attenuation(N, x, y, tc, bSE)` with `bSE = γ·G·√(D_rest·tc)`.

---

## Variables

| Symbol | Code name | Type | Value / bounds |
|--------|-----------|------|----------------|
| γ | `_GYRO` | Physical constant | 267.522 ms⁻¹ mT⁻¹ |
| D₀ | `D0_FIXED` | **Fixed** | 3.2×10⁻¹² m²/ms |
| RN | `RN_FIXED` | **Fixed or configurable** | see per-notebook config |
| tc | `tc` (→ `log(tc)` optimised) | **Fitted or pinned** | [0.5, 500] ms |
| α | `alpha` | **Fitted or pinned** | [0, 1] |
| M0 | `M0` | **Fitted or pinned** | ≥ 0 |
| C | `C` | **Fitted or pinned** | (−∞, +∞) |

**Configuration flags** — each `_FIXED` variable follows the same convention: `None` means the
parameter is free (optimised); a float value pins it. Not all flags exist in every notebook:

| Flag | `_global` | `_joint` | `_mixed` (new) | `_free` |
|------|-----------|----------|----------------|---------|
| `TC_FIXED` | — | — | ✓ | — |
| `ALPHA_FIXED` | ✓ | ✓ | ✓ | ✓ |
| `M0_FIXED` | — | ✓ | ✓ | ✓ |
| `C_FIXED` | — | — | ✓ | — |
| `RN_FIXED` | scalar | scalar | ✓ | scalar |

`—` means the parameter is always free (tc, C in global/joint/free) or always fixed (RN in global/joint/free).
`scalar` means the variable exists but only accepts a float, not `None`.

**Optimizer**: `scipy.optimize.least_squares`, method TRF, max 10 000 function evaluations.
`tc` is optimised in log-space (`params[0] = log(tc)`) to enforce positivity and improve
conditioning over many orders of magnitude.

---

## Input data

All four notebooks load the same rows from `master.long.parquet`:

```
row_kind == "signal_rotated"
direction ∈ {tra, long}        (DIRECTIONS)
N ∈ {4, 8}                     (N_LIST)
stat == "avg"
```

The effective gradient is:

```
G_eff = g_thorsten × grad_correction_factor
```

Fitting groups are `(subj, roi, direction)`.

---

## Four fitting strategies

### 1. `fit_ogse_mixed_global.ipynb` — Global fit (all td jointly)

**Fitting unit**: one `(subj, roi, direction)` group.

The residual vector concatenates **every** `(td, N)` curve in the group:

```
residuals = [
    model(G, td=t₁, N=8) − data(td=t₁, N=8),
    model(G, td=t₁, N=4) − data(td=t₁, N=4),
    model(G, td=t₂, N=8) − data(td=t₂, N=8),
    model(G, td=t₂, N=4) − data(td=t₂, N=4),
    ...
]
```

**Parameter scope**:

| Parameter | Scope |
|-----------|-------|
| tc | Global — one value for all td and N |
| α | Global — one value for all td and N (or pinned) |
| M0 | Per-td — shared between N=4 and N=8 at the same td |
| C | Per-td — shared between N=4 and N=8 at the same td |
| D0, RN | Fixed |

**Output directory**: `analysis/brains/ogse_experiments/mixed_global_fits/`

---

### 2. `fit_ogse_mixed_joint.ipynb` — Per-td fit (N=4 and N=8 jointly)

**Fitting unit**: one `(subj, roi, direction, td)` group.

The residual vector concatenates N=4 and N=8 for that single td:

```
residuals = [
    model(G, td, N=4) − data(td, N=4),
    model(G, td, N=8) − data(td, N=8),
]
```

**Parameter scope**:

| Parameter | Scope |
|-----------|-------|
| tc | Per-td — one value per td |
| α | Per-td — one value per td (or pinned) |
| M0 | Per-td — shared between N=4 and N=8 |
| C | Per-td — shared between N=4 and N=8 |
| D0, RN | Fixed scalar |

**Output directory**: `analysis/brains/ogse_experiments/mixed_joint_fits/`

---

### 3. `fit_ogse_mixed.ipynb` — Per-curve fit, all parameters configurable

**Fitting unit**: one `(subj, roi, direction, td, N)` curve.

Each curve is fitted on its own. Every parameter has an individual `_FIXED` flag;
set it to `None` to fit it or to a float to pin it.

**Parameter scope**:

| Parameter | Scope | Config flag |
|-----------|-------|-------------|
| tc | Per-curve (or pinned) | `TC_FIXED` |
| α | Per-curve (or pinned) | `ALPHA_FIXED` |
| M0 | Per-curve (or pinned) | `M0_FIXED` |
| C | Per-curve (or pinned) | `C_FIXED` |
| RN | Per-curve (or pinned) | `RN_FIXED` |
| D0 | Fixed scalar | — |

**Output directory**: `analysis/brains/ogse_experiments/mixed_fits/`

---

### 4. `fit_ogse_mixed_free.ipynb` — Free fit (each curve independently, limited config)

**Fitting unit**: one `(subj, roi, direction, td, N)` curve.

Each curve is fitted on its own. tc and C are always free; RN is always a fixed scalar.

**Parameter scope**:

| Parameter | Scope | Config flag |
|-----------|-------|-------------|
| tc | Per-curve — always fitted | — |
| α | Per-curve (or pinned) | `ALPHA_FIXED` |
| M0 | Per-curve (or pinned) | `M0_FIXED` |
| C | Per-curve — always fitted | — |
| D0, RN | Fixed scalar | — |

**Output directory**: `analysis/brains/ogse_experiments/mixed_free_fits/`

---

## Output files (common to all four notebooks)

| File | Content |
|------|---------|
| `fit_results.xlsx` | One row per `(subj, roi, dir, td, N)` with all fitted parameters |
| `fit_{subj}_{roi}_{dir}.png` | Data + fitted curves in absolute signal units |
| `fit_norm_{subj}_{roi}_{dir}.png` | Same, normalised by M0 (M0 = 1 in model, data divided by M0) |
| `tc_vs_td_{dir}.png` | Fitted tc versus td for all subjects and ROIs |

---

## Reproducing the normalised curve from `fit_results.xlsx`

```python
y_norm = M_ogse_mixed_offset(
    td_ms, G_eff, N, td_ms / N,
    tc_ms, alpha, M0=1.0, D0_m2ms, C / M0, RN / M0
)
# corresponding normalised data: value / M0
```
