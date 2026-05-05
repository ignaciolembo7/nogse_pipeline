### Stage 6: Tensor-based rotation of brain OGSE signals

**What goes in**

- long-form brain signal tables with six diffusion directions

**What happens conceptually**

- for each ROI and `b`-step, the pipeline estimates a diffusion tensor `D` from the directional attenuations;
- the tensor is diagonalized to obtain its principal axes;
- the signal is re-expressed along fixed axes such as:
  - tensor eigenvectors `eig1`, `eig2`, `eig3`
  - laboratory axes `x`, `y`, `z`
  - combined axes `long` and `tra`
- it also writes `D_proj`, the projection of the tensor along each rotated axis.

**What comes out**

- rotated signal tables
- projected diffusivity tables (`*.Dproj.long.parquet`)

**Why this step is needed**

- the raw six-direction measurements are hard to compare directly across subjects;
- rotation reduces them to axes that better reflect underlying anisotropy, especially for structures such as the corpus callosum.

**Key physical or mathematical idea**

The tensor fit uses the standard relation:

```text
-log(S/S0) / b = n^T D n
```

and then projects the fitted tensor onto chosen directions:

```text
D_proj = n^T D n
```

**Code**

- `scripts/rotate_ogse_tensor.py`
- `src/signal_rotation/rotation_tensor.py`
  - `fit_tensor_from_signals`
  - `D_proj`
  - `rotate_signals_tensor`