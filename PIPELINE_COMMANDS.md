# Pipeline commands

Run from the project root. El modelo default para OGSE es `ogse_mixed_offset`.

---

## Cómo funcionan las variables entre pasos

Cuando pasás múltiples pasos en un mismo comando (ej. `run_dataset.sh brain ogse ingest rotate ...`), el runner los ejecuta en secuencia dentro del mismo proceso. Cada paso corre en un **subshell aislado** (`bash step_script.sh`), así que variables que un paso setea internamente NO se propagan al siguiente.

Lo que SÍ se propaga:
- Cualquier variable que seteás **antes** de llamar a `run_dataset.sh` (en el entorno externo) la heredan TODOS los pasos.
- `MASTER_PARQUET` y `MASTER_LAST_POINTS_APPLIED` se exportan en el proceso padre automáticamente cuando usás `MASTER_LAST_POINTS_BY_TD`, así que el filtrado se aplica a todos los pasos en secuencia sin repetir el filtro.

Lo que NO se propaga:
- `SIGNAL_FIT_OUT_ROOT` es calculado dentro del subshell del paso `fit_signal` y no vuelve al proceso padre. Por eso hay que pasarlo explícito en el entorno externo para que `contrast_resampled` lo pueda ver.

Conflictos entre pasos: no hay, porque cada paso corre en subshell aislado. Si dos pasos usan la misma variable (ej. `MODEL`), cada uno la lee del entorno externo y la usa de forma independiente.

---

## Sin filtrar — brains (OGSE)

### Paso a paso (correr en orden, esperar que cada uno termine)

```
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest > logs/01_ingest.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate > logs/02_rotate.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_signal > logs/03_plot_signal.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse grad_correction > logs/05_grad_correction.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal > logs/04_fit_signal.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_monoexp_d > logs/06b_plot_monoexp_d.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse alpha > logs/06_alpha.log 2>&1 &
GLOBAL_SIGNAL_MODEL=ogse_mixed_offset GLOBAL_SIGNAL_G_TYPE=g_thorsten GLOBAL_SIGNAL_YCOL=value GLOBAL_SIGNAL_ROW_KIND=signal_rotated GLOBAL_SIGNAL_STAT=avg GLOBAL_SIGNAL_MIN_POINTS=4 GLOBAL_SIGNAL_DIRECTIONS="long tra" GLOBAL_SIGNAL_ROIS=ALL GLOBAL_SIGNAL_SUBJS=ALL GLOBAL_SIGNAL_TC_MODE=global_td GLOBAL_SIGNAL_ALPHA_MODE=global_td GLOBAL_SIGNAL_RN_MODE=fixed GLOBAL_SIGNAL_RN_FIXED=15 GLOBAL_SIGNAL_M0_MODE=global_contrast GLOBAL_SIGNAL_C_MODE=global_contrast GLOBAL_SIGNAL_D0_MODE=fixed GLOBAL_SIGNAL_D0_FIXED=3.2e-12 GLOBAL_SIGNAL_APPLY_GRAD_CORR=true GLOBAL_SIGNAL_MANIFEST=nogse_pipeline/bash_template/manifests/brains_ogse/contrasts.csv GLOBAL_SIGNAL_OUT_ROOT=analysis/brains/ogse_experiments/fits/ogse_signal_ogse_mixed_offset nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_global_signal > logs/04b_fit_global_signal.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse contrast > logs/07_contrast.log 2>&1 &
SIGNAL_FIT_OUT_ROOT=analysis/brains/ogse_experiments/fits/master/ogse_value_norm_vs_bvaluethorsten_ogse_mixed_offset nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse contrast_resampled > logs/07b_contrast_resampled.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast > logs/08_fit_contrast.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse extract_tc_peak > logs/09_extract_tc_peak.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse tc > logs/10_tc.log 2>&1 &
```

### Todo en uno

```
SIGNAL_FIT_OUT_ROOT=analysis/brains/ogse_experiments/fits/master/ogse_value_norm_vs_bvaluethorsten_ogse_mixed_offset nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest rotate grad_correction fit_signal contrast contrast_resampled fit_contrast extract_tc_peak tc > logs/brain_full.log 2>&1 &
```

---

## Sin filtrar — phantoms (OGSE)

### Paso a paso

```
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse ingest > logs/01_ingest.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse rotate > logs/02_rotate.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse plot_signal > logs/03_plot_signal.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse grad_correction > logs/05_grad_correction.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse fit_signal > logs/04_fit_signal.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse plot_monoexp_d > logs/06b_plot_monoexp_d.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse alpha > logs/06_alpha.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse contrast > logs/07_contrast.log 2>&1 &
SIGNAL_FIT_OUT_ROOT=analysis/phantoms/ogse_experiments/fits/master/ogse_value_norm_vs_bvaluethorsten_ogse_mixed_offset nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse contrast_resampled > logs/07b_contrast_resampled.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse fit_contrast > logs/08_fit_contrast.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse extract_tc_peak > logs/09_extract_tc_peak.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse tc > logs/10_tc.log 2>&1 &
```

### Todo en uno

```
SIGNAL_FIT_OUT_ROOT=analysis/phantoms/ogse_experiments/fits/master/ogse_value_norm_vs_bvaluethorsten_ogse_mixed_offset nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse ingest rotate grad_correction fit_signal contrast contrast_resampled fit_contrast extract_tc_peak tc > logs/phantom_full.log 2>&1 &
```

---

## Con filtrar últimos N puntos por td y N — brains (OGSE)

Reemplazá `120:8=6,120:4=4,210=8` con tus valores. Formato: `td=puntos` (todos los N de ese td) o `td:N=puntos` (td y N específicos), separados por coma.

### Paso a paso

```
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest > logs/01_ingest.log 2>&1 &
MASTER_LAST_POINTS_BY_TD="120:8=6,120:4=4,210=8" nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse filter_master_points > logs/00_filter.log 2>&1 &
MASTER_PARQUET=analysis/brains/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse rotate > logs/02_rotate.log 2>&1 &
MASTER_PARQUET=analysis/brains/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_signal > logs/03_plot_signal.log 2>&1 &
MASTER_PARQUET=analysis/brains/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse grad_correction > logs/05_grad_correction.log 2>&1 &
MASTER_PARQUET=analysis/brains/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal > logs/04_fit_signal.log 2>&1 &
MASTER_PARQUET=analysis/brains/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_monoexp_d > logs/06b_plot_monoexp_d.log 2>&1 &
MASTER_PARQUET=analysis/brains/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse alpha > logs/06_alpha.log 2>&1 &
MASTER_PARQUET=analysis/brains/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse contrast > logs/07_contrast.log 2>&1 &
MASTER_PARQUET=analysis/brains/ogse_experiments/master.last_points.long.parquet SIGNAL_FIT_OUT_ROOT=analysis/brains/ogse_experiments/fits/master.last_points/ogse_value_norm_vs_bvaluethorsten_ogse_mixed_offset nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse contrast_resampled > logs/07b_contrast_resampled.log 2>&1 &
MASTER_PARQUET=analysis/brains/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast > logs/08_fit_contrast.log 2>&1 &
MASTER_PARQUET=analysis/brains/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse extract_tc_peak > logs/09_extract_tc_peak.log 2>&1 &
MASTER_PARQUET=analysis/brains/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse tc > logs/10_tc.log 2>&1 &
```

### Todo en uno

```
MASTER_LAST_POINTS_BY_TD="120:8=6,120:4=4,210=8" SIGNAL_FIT_OUT_ROOT=analysis/brains/ogse_experiments/fits/master.last_points/ogse_value_norm_vs_bvaluethorsten_ogse_mixed_offset nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse ingest rotate grad_correction fit_signal contrast contrast_resampled fit_contrast extract_tc_peak tc > logs/brain_filtered.log 2>&1 &
```

> En el modo "todo en uno" con filtrado, `MASTER_LAST_POINTS_BY_TD` dispara el filtrado automáticamente antes de cada paso (el pipeline lo detecta y aplica la primera vez, luego no lo repite).

---

## Con filtrar últimos N puntos por td y N — phantoms (OGSE)

### Paso a paso

```
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse ingest > logs/01_ingest.log 2>&1 &
MASTER_LAST_POINTS_BY_TD="120:8=6,120:4=4,210=8" nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse filter_master_points > logs/00_filter.log 2>&1 &
MASTER_PARQUET=analysis/phantoms/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse rotate > logs/02_rotate.log 2>&1 &
MASTER_PARQUET=analysis/phantoms/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse plot_signal > logs/03_plot_signal.log 2>&1 &
MASTER_PARQUET=analysis/phantoms/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse grad_correction > logs/05_grad_correction.log 2>&1 &
MASTER_PARQUET=analysis/phantoms/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse fit_signal > logs/04_fit_signal.log 2>&1 &
MASTER_PARQUET=analysis/phantoms/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse plot_monoexp_d > logs/06b_plot_monoexp_d.log 2>&1 &
MASTER_PARQUET=analysis/phantoms/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse alpha > logs/06_alpha.log 2>&1 &
MASTER_PARQUET=analysis/phantoms/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse contrast > logs/07_contrast.log 2>&1 &
MASTER_PARQUET=analysis/phantoms/ogse_experiments/master.last_points.long.parquet SIGNAL_FIT_OUT_ROOT=analysis/phantoms/ogse_experiments/fits/master.last_points/ogse_value_norm_vs_bvaluethorsten_ogse_mixed_offset nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse contrast_resampled > logs/07b_contrast_resampled.log 2>&1 &
MASTER_PARQUET=analysis/phantoms/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse fit_contrast > logs/08_fit_contrast.log 2>&1 &
MASTER_PARQUET=analysis/phantoms/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse extract_tc_peak > logs/09_extract_tc_peak.log 2>&1 &
MASTER_PARQUET=analysis/phantoms/ogse_experiments/master.last_points.long.parquet nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse tc > logs/10_tc.log 2>&1 &
```

### Todo en uno

```
MASTER_LAST_POINTS_BY_TD="120:8=6,120:4=4,210=8" SIGNAL_FIT_OUT_ROOT=analysis/phantoms/ogse_experiments/fits/master.last_points/ogse_value_norm_vs_bvaluethorsten_ogse_mixed_offset nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse ingest rotate grad_correction fit_signal contrast contrast_resampled fit_contrast extract_tc_peak tc > logs/phantom_filtered.log 2>&1 &
```

---

## NOGSE — brains

> **Antes de correr:** llenar `manifests/brains_nogse/signal_fits.csv` y `manifests/brains_nogse/contrasts.csv` con los parámetros reales de tus sesiones NOGSE (ver comentarios en esos archivos).

Diferencias respecto a OGSE:
- Sin `grad_correction` (no hay manifest para NOGSE)
- Modelo default: `nogse_free` (g_type: `g`, no `bvalue_thorsten`)
- `SIGNAL_FIT_OUT_ROOT` con path distinto

### Paso a paso

```
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain nogse ingest > logs/brain_nogse_01_ingest.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain nogse rotate > logs/brain_nogse_02_rotate.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain nogse plot_signal > logs/brain_nogse_03_plot_signal.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain nogse fit_signal > logs/brain_nogse_04_fit_signal.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain nogse alpha > logs/brain_nogse_06_alpha.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain nogse contrast > logs/brain_nogse_07_contrast.log 2>&1 &
SIGNAL_FIT_OUT_ROOT=analysis/brains/nogse_experiments/fits/master/nogse_value_norm_vs_g_nogse_free nohup bash nogse_pipeline/bash_template/run_dataset.sh brain nogse contrast_resampled > logs/brain_nogse_07b_contrast_resampled.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain nogse fit_contrast > logs/brain_nogse_08_fit_contrast.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain nogse extract_tc_peak > logs/brain_nogse_09_extract_tc_peak.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain nogse tc > logs/brain_nogse_10_tc.log 2>&1 &
```

### Todo en uno

```
SIGNAL_FIT_OUT_ROOT=analysis/brains/nogse_experiments/fits/master/nogse_value_norm_vs_g_nogse_free nohup bash nogse_pipeline/bash_template/run_dataset.sh brain nogse ingest rotate fit_signal contrast contrast_resampled fit_contrast extract_tc_peak tc > logs/brain_nogse_full.log 2>&1 &
```

### Con filtrar últimos N puntos por td y N

```
MASTER_LAST_POINTS_BY_TD="120:8=6,120:4=4,210=8" SIGNAL_FIT_OUT_ROOT=analysis/brains/nogse_experiments/fits/master.last_points/nogse_value_norm_vs_g_nogse_free nohup bash nogse_pipeline/bash_template/run_dataset.sh brain nogse ingest rotate fit_signal contrast contrast_resampled fit_contrast extract_tc_peak tc > logs/brain_nogse_filtered.log 2>&1 &
```

---

## NOGSE — phantoms

> **Antes de correr:** llenar `manifests/phantoms_nogse/signal_fits.csv` y `manifests/phantoms_nogse/contrasts.csv`.

### Paso a paso

```
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse ingest > logs/phantom_nogse_01_ingest.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse rotate > logs/phantom_nogse_02_rotate.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse plot_signal > logs/phantom_nogse_03_plot_signal.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse fit_signal > logs/phantom_nogse_04_fit_signal.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse alpha > logs/phantom_nogse_06_alpha.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse contrast > logs/phantom_nogse_07_contrast.log 2>&1 &
SIGNAL_FIT_OUT_ROOT=analysis/phantoms/nogse_experiments/fits/master/nogse_value_norm_vs_g_nogse_free nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse contrast_resampled > logs/phantom_nogse_07b_contrast_resampled.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse fit_contrast > logs/phantom_nogse_08_fit_contrast.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse extract_tc_peak > logs/phantom_nogse_09_extract_tc_peak.log 2>&1 &
nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse tc > logs/phantom_nogse_10_tc.log 2>&1 &
```

### Todo en uno

```
SIGNAL_FIT_OUT_ROOT=analysis/phantoms/nogse_experiments/fits/master/nogse_value_norm_vs_g_nogse_free nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse ingest rotate fit_signal contrast contrast_resampled fit_contrast extract_tc_peak tc > logs/phantom_nogse_full.log 2>&1 &
```

### Con filtrar últimos N puntos por td y N

```
MASTER_LAST_POINTS_BY_TD="120:8=6,120:4=4,210=8" SIGNAL_FIT_OUT_ROOT=analysis/phantoms/nogse_experiments/fits/master.last_points/nogse_value_norm_vs_g_nogse_free nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom nogse ingest rotate fit_signal contrast contrast_resampled fit_contrast extract_tc_peak tc > logs/phantom_nogse_filtered.log 2>&1 &
```

---

## Preprocessing

Los scripts de preprocessing se corren **directamente** (no vía `run_dataset.sh`). Correr en el orden indicado, esperar que cada uno termine antes del siguiente.

### Brains — OGSE (DICOM → NIfTI → extracción de señal)

```
nohup bash nogse_pipeline/bash_template/steps/preprocessing/brains_ogse/0.0-run_dicom2nifti.sh > logs/pre_brains_ogse_dicom.log 2>&1 &
nohup bash nogse_pipeline/bash_template/steps/preprocessing/brains_ogse/1.0-run_BRAINS-denoised_topup_signal_extraction.sh > logs/pre_brains_ogse_extract.log 2>&1 &
```

### Phantoms — OGSE (DICOM → NIfTI → gval/gvec → b0 → masks → extracción)

```
nohup bash nogse_pipeline/bash_template/steps/preprocessing/phantoms_ogse/0.0-run_dicom2nifti.sh > logs/pre_phantoms_ogse_dicom.log 2>&1 &
nohup bash nogse_pipeline/bash_template/steps/preprocessing/phantoms_ogse/0.1-run_make_gval_gvec.sh > logs/pre_phantoms_ogse_gvec.log 2>&1 &
nohup bash nogse_pipeline/bash_template/steps/preprocessing/phantoms_ogse/0.2-prep_phantom_b0.sh > logs/pre_phantoms_ogse_b0.log 2>&1 &
nohup bash nogse_pipeline/bash_template/steps/preprocessing/phantoms_ogse/0.3-copy_selected_files.sh > logs/pre_phantoms_ogse_copy.log 2>&1 &
nohup bash nogse_pipeline/bash_template/steps/preprocessing/phantoms_ogse/1.0-run_PHANTOM-denoised_signal_extraction.sh > logs/pre_phantoms_ogse_extract.log 2>&1 &
```

### Phantoms — NOGSE (mismo orden que OGSE)

```
nohup bash nogse_pipeline/bash_template/steps/preprocessing/phantoms_nogse/0.0-run_dicom2nifti.sh > logs/pre_phantoms_nogse_dicom.log 2>&1 &
nohup bash nogse_pipeline/bash_template/steps/preprocessing/phantoms_nogse/0.1-run_make_gval_gvec.sh > logs/pre_phantoms_nogse_gvec.log 2>&1 &
nohup bash nogse_pipeline/bash_template/steps/preprocessing/phantoms_nogse/0.2-prep_phantom_b0.sh > logs/pre_phantoms_nogse_b0.log 2>&1 &
nohup bash nogse_pipeline/bash_template/steps/preprocessing/phantoms_nogse/0.3-copy_selected_files.sh > logs/pre_phantoms_nogse_copy.log 2>&1 &
nohup bash nogse_pipeline/bash_template/steps/preprocessing/phantoms_nogse/1.0-run_PHANTOM-denoised_signal_extraction.sh > logs/pre_phantoms_nogse_extract.log 2>&1 &
```

> **Nota:** No existe `preprocessing/brains_nogse/` todavía. Si tenés sesiones NOGSE de brains, hay que crear esa carpeta siguiendo el patrón de `brains_ogse/`.

### Parámetros DICOM (independiente del tipo de secuencia)

```
nohup bash nogse_pipeline/bash_template/steps/preprocessing/dicom_params/0.0-run_extract_dicom_sequence_metadata.sh > logs/dicom_params.log 2>&1 &
```

### Exportar tabla master como .xslx (para visualizacion)

```
MASTER_PARQUET=analysis/brains/ogse_experiments/master.long.parquet MASTER_XLSX=analysis/brains/ogse_experiments/master.xlsx nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse export_master_xlsx > logs/export_master_xlsx.log 2>&1 &
```

```
MASTER_PARQUET=analysis/phantoms/ogse_experiments/master.long.parquet MASTER_XLSX=analysis/phantoms/ogse_experiments/master.xlsx nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse export_master_xlsx > logs/export_master_xlsx.log 2>&1 &
```

---

## Referencia de pasos adicionales

### `plot_signal` — graficar curvas de señal

Genera plots de señal vs gradiente directamente del `master.long.parquet`.

| Variable | Default | Descripción |
|---|---|---|
| `PLOT_OUT_ROOT` | `$ANALYSIS_ROOT/plots-master/signal` | Directorio de salida |
| `PLOT_ROW_KIND` | `signal_rotated` | `signal_rotated` o `signal` |
| `PLOT_SIGNAL_YCOL` | `value_norm` | `value` o `value_norm` |
| `PLOT_SIGNAL_XCOL` | (según tipo_seq) | Columna del eje X (ej. `g_thorsten`, `g`, `bvalue_thorsten`) |
| `PLOT_STAT` | `avg` | `avg` o `std` |
| `PLOT_SUBJ` | (todos) | Filtro de sujeto |
| `PLOT_ROI` | (todos) | Filtro de ROI |
| `PLOT_DIRECTION` | (todas) | Filtro de dirección (ej. `long`, `tra`) |
| `PLOT_TD_MS` | (todos) | Filtro de td_ms |
| `PLOT_N` | (todos) | Filtro de N |
| `PLOT_SIGNAL_EXTRA_ARGS` | — | Args extra para `plot_*_signal_vs_g.py` |

```
# Plot señal para un ROI y dirección específica
PLOT_ROI=Left-Lateral-Ventricle PLOT_DIRECTION=long \
  nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_signal > logs/plot_signal.log 2>&1 &

# Plot normalizado con g_thorsten como eje X
PLOT_SUBJ=20220622_BRAIN PLOT_DIRECTION=long PLOT_SIGNAL_XCOL=g_thorsten \
  nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_signal > logs/plot_signal.log 2>&1 &
```

---

### `plot_monoexp_d` — graficar D monoexp vs tiempo

Requiere haber corrido `fit_signal` antes (usa los parquets de fits de señal).

| Variable | Default | Descripción |
|---|---|---|
| `SIGNAL_FITS_ROOT` | `$ANALYSIS_ROOT/fits/<master>/<experiment>_<model>` | Root escaneado para fits de señal |
| `MONOEXP_D_OUT_DIR` | `$ANALYSIS_ROOT/plots-master/monoexp_D_vs_time` | Directorio de salida |
| `PLOT_MONOEXP_D_EXTRA_ARGS` | — | Args extra para `plot_monoexp_D_vs_time.py` |

```
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_monoexp_d > logs/plot_monoexp_d.log 2>&1 &

# Con root explícito
SIGNAL_FITS_ROOT=analysis/brains/ogse_experiments/fits/master/ogse_value_norm_vs_bvaluethorsten_monoexp \
  nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse plot_monoexp_d > logs/plot_monoexp_d.log 2>&1 &
```

---

### `fit_signal_gradcorr` — fit de señal con corrección de gradiente

Equivale a `fit_signal` con `--apply_grad_corr` activo en todos los fits. Usa el mismo manifest `signal_fits.csv` y las mismas variables que `fit_signal`, pero requiere haber corrido `grad_correction` antes.

```
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_signal_gradcorr > logs/fit_signal_gradcorr.log 2>&1 &
```

---

### `fit_global_signal` — fit de señal mixto/global

Ajusta modelos globales (parámetros compartidos entre curvas) directamente sobre el `master.long.parquet`.

| Variable | Default | Descripción |
|---|---|---|
| `GLOBAL_SIGNAL_MODEL` | `ogse_mixed_offset` (ogse) / `nogse_mixed_offset` (nogse) | Modelo a ajustar |
| `GLOBAL_SIGNAL_OUT_ROOT` | `$ANALYSIS_ROOT/fits/ogse_signal_<model>` | Directorio de salida |
| `GLOBAL_SIGNAL_ROW_KIND` | `signal_rotated` | `signal_rotated` o `signal` |
| `GLOBAL_SIGNAL_YCOL` | `value` | `value` o `value_norm` |
| `GLOBAL_SIGNAL_G_TYPE` | (según tipo_seq) | Columna de gradiente (ej. `g_thorsten`) |
| `GLOBAL_SIGNAL_STAT` | `avg` | `avg` o `std` |
| `GLOBAL_SIGNAL_MIN_POINTS` | `4` | Puntos mínimos por grupo para intentar fit |
| `GLOBAL_SIGNAL_TC_MODE` | `global_td` | `fixed\|free\|global_td\|global_contrast` |
| `GLOBAL_SIGNAL_TC_FIXED` | — | Valor fijo de tc [ms] (cuando TC_MODE=fixed) |
| `GLOBAL_SIGNAL_ALPHA_MODE` | `global_td` | `fixed\|free\|global_td\|global_contrast` |
| `GLOBAL_SIGNAL_ALPHA_FIXED` | — | Valor fijo de alpha (cuando ALPHA_MODE=fixed) |
| `GLOBAL_SIGNAL_RN_MODE` | `global_td` | `fixed\|free\|global_td\|global_contrast` |
| `GLOBAL_SIGNAL_RN_FIXED` | — | Valor fijo de RN (cuando RN_MODE=fixed) |
| `GLOBAL_SIGNAL_M0_MODE` | `global_contrast` | `fixed\|free\|global_td\|global_contrast` |
| `GLOBAL_SIGNAL_M0_FIXED` | — | Valor fijo de M0 (cuando M0_MODE=fixed) |
| `GLOBAL_SIGNAL_C_MODE` | `global_contrast` | `fixed\|free\|global_td\|global_contrast` |
| `GLOBAL_SIGNAL_C_FIXED` | — | Valor fijo de C (cuando C_MODE=fixed) |
| `GLOBAL_SIGNAL_D0_MODE` | `fixed` | `fixed\|free\|global_td\|global_contrast` |
| `GLOBAL_SIGNAL_D0_FIXED` | `3.2e-12` (brain) / `2.3e-12` (phantom) | D0 fijo en m²/s |
| `GLOBAL_SIGNAL_DIRECTIONS` | `ALL` | Direcciones a incluir (ej. `"long tra"`) |
| `GLOBAL_SIGNAL_ROIS` | `ALL` | ROIs a incluir |
| `GLOBAL_SIGNAL_SUBJS` | `ALL` | Sujetos a incluir |
| `GLOBAL_SIGNAL_APPLY_GRAD_CORR` | `true` | Aplicar corrección de gradiente |
| `GLOBAL_SIGNAL_MANIFEST` | `contrasts.csv` (cuando M0/C son global_contrast) | Manifest con curvas a ajustar |
| `GLOBAL_SIGNAL_EXTRA_ARGS` | — | Args extra para `fit_global_signal.py` |

```
# Fit estándar (ogse_mixed_offset, D0 fijo, tc/alpha global por td)
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_global_signal > logs/fit_global_signal.log 2>&1 &

# Con RN fijo
GLOBAL_SIGNAL_RN_MODE=fixed GLOBAL_SIGNAL_RN_FIXED=10 \
  nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_global_signal > logs/fit_global_signal.log 2>&1 &

# Modelo alternativo, sin corrección de gradiente
GLOBAL_SIGNAL_MODEL=ogse_rest GLOBAL_SIGNAL_APPLY_GRAD_CORR=false \
GLOBAL_SIGNAL_OUT_ROOT=analysis/brains/ogse_experiments/fits/ogse_signal_ogse_rest_raw \
  nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_global_signal > logs/fit_global_signal.log 2>&1 &

# Solo para un subset de direcciones y ROIs
GLOBAL_SIGNAL_DIRECTIONS="long tra" GLOBAL_SIGNAL_ROIS="Left-Lateral-Ventricle AntCC" \
  nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_global_signal > logs/fit_global_signal.log 2>&1 &
```

Modelos disponibles (OGSE): `monoexp`, `free_ogse`, `ogse_mixed_offset`, `ogse_mixed_global`, `rest`, `rest_offset`, `ogse_free`, `ogse_rest`, `ogse_rest_offset`

---

### `fit_contrast` / `fit_contrast_free` / `fit_contrast_mixed_global` — fit de contraste

Ajusta modelos analíticos sobre la tabla de contraste (`contrast_resampled`). Requiere haber corrido `contrast_resampled` antes.

| Variable | Default | Descripción |
|---|---|---|
| `CONTRAST_PARQUET` | `$ANALYSIS_ROOT/contrast-data-master/master_contrast.parquet` | Tabla de contraste a ajustar |
| `FIT_OUT_ROOT` | `$ANALYSIS_ROOT/fits/<master>/<type_seq>_<ycol>_vs_<gtype>_<model>` | Directorio de salida |
| `FIT_MODEL` | `ogse_free` (ogse) / `nogse_free` (nogse) | Modelo de ajuste |
| `FIT_GBASE` | (según tipo_seq) | Columna de gradiente para el eje X |
| `FIT_YCOL` | `value_norm` | `value` o `value_norm` |
| `FIT_EXTRA_ARGS` | — | Args extra para `fit_*_contrast_vs_g.py` |

Los presets:
- `fit_contrast_free` → setea `FIT_MODEL=ogse_free` (OGSE) o `FIT_MODEL=nogse_free` (NOGSE) si no está seteado
- `fit_contrast_mixed_global` → setea `FIT_MODEL=ogse_mixed_global` si no está seteado

```
# Fit estándar (ogse_free por default)
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast > logs/fit_contrast.log 2>&1 &

# Preset free
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast_free > logs/fit_contrast_free.log 2>&1 &

# Preset mixed_global
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast_mixed_global > logs/fit_contrast_mixed_global.log 2>&1 &

# Con modelo y opciones explícitas
FIT_MODEL=ogse_mixed FIT_GBASE=g_lin_max FIT_YCOL=value_norm \
  nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast > logs/fit_contrast.log 2>&1 &

# Con D0 fijo y bounds de tc
FIT_EXTRA_ARGS="--fix_D0 3.2e-12 --tc_bounds 0.5 200.0" \
  nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast > logs/fit_contrast.log 2>&1 &

# Parámetros globales (tc compartido entre curvas)
FIT_EXTRA_ARGS="--global_params tc_ms" \
  nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse fit_contrast > logs/fit_contrast.log 2>&1 &
```

---

### `alpha` — calcular alpha macro (D vs Delta)

Calcula alpha macroscópico a partir de los fits de señal (`fit_signal` o `fit_global_signal`). También genera plots de D vs Delta_app.

| Variable | Default | Descripción |
|---|---|---|
| `ALPHA_N` | `1` | Selector de N para `make_alpha_macro_summary.py` |
| `ALPHA_OUT_DIR` | `$ANALYSIS_ROOT/alpha_macro/master` | Directorio de salida |
| `ALPHA_EXTRA_ARGS` | — | Args extra para `make_alpha_macro_summary.py` |
| `DPROJ_N` | (igual a ALPHA_N) | Selector de N para plots D vs Delta (si difiere) |
| `DPROJ_DIRS` | — | Filtro de direcciones para D-projection |
| `DPROJ_ROIS` | — | Filtro de ROIs para D-projection |
| `PLOT_D0_EXTRA_ARGS` | — | Args extra solo para `plot_D0_vs_Delta.py` |

Args útiles en `ALPHA_EXTRA_ARGS`:
- `--bvalmax 5` — límite de b-value máximo para el fit de alpha
- `--roi-bvalmax AntCC=7` — límite distinto por ROI
- `--dirs long tra` — restringir a esas direcciones

Salidas en `$ALPHA_OUT_DIR/`:
- `summary_alpha_values.xlsx` — alpha_macro por sujeto/ROI/dirección
- `D_vs_delta_app.combined.xlsx` — D vs Delta_app agregado
- `alpha_macro_vs_roi.png` — plot resumen por ROI
- `<subj>/<roi>/<dir>_*.png` — curvas individuales

```
# Alpha estándar para brains (N=1)
ALPHA_N=1 \
ALPHA_EXTRA_ARGS="--bvalmax 5 --roi-bvalmax AntCC=7 --roi-bvalmax CSF=3 --dirs long tra" \
  nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse alpha > logs/alpha.log 2>&1 &

# Alpha para phantoms
ALPHA_EXTRA_ARGS="--bvalmax 5 --roi-bvalmax Syringe=7" \
  nohup bash nogse_pipeline/bash_template/run_dataset.sh phantom ogse alpha > logs/alpha.log 2>&1 &
```

El archivo `summary_alpha_values.xlsx` producido aquí es el que se pasa a `tc` con `--summary-alpha` cuando se usa `TC_METHOD=pseudohuber_fixed_macro`.

---

### `extract_tc_peak` — consolidar fits y extraer tabla tc_peak

Consolida los `fit_params` de `fit_contrast` (paso 08) en una tabla canónica `tc_peak` y genera paneles de visualización.

| Variable | Default | Descripción |
|---|---|---|
| `FIT_OUT_ROOT` | derivado automáticamente de FIT_MODEL/FIT_GBASE/FIT_YCOL | Directorio con los fits de contraste |
| `TC_PEAK_DIR` | `$FIT_OUT_ROOT/tc_peak` | Directorio de salida de la tabla tc_peak |
| `TC_PEAK_MODELS` | (todos) | Filtro de modelos |
| `TC_PEAK_SUBJS` | (todos) | Filtro de sujetos |
| `TC_PEAK_ROIS` | (todos) | Filtro de ROIs |
| `TC_PEAK_DIRECTIONS` | (todas) | Filtro de direcciones |
| `TC_PEAKS_CONTRAST_ROOT` | (derivado de contrast_resampled) | Root con tablas de contraste resampleado |
| `TC_PEAKS_CONTRAST_SOURCE` | `fitted_resampled` | `direct\|fitted_resampled\|auto` |
| `TC_PEAKS_PEAK_SOURCE` | `resampled` | `standard\|resampled\|both` |
| `TC_PEAKS_X_VARS` | `g Ld lcf lcf_a tc` | Variables del eje X en los paneles |
| `TC_PEAKS_MODELS` | (todos) | Filtro de modelos para paneles |
| `TC_PEAKS_SUBJS` | (todos) | Filtro de sujetos para paneles |
| `TC_PEAKS_ROIS` | (todos) | Filtro de ROIs para paneles |
| `TC_PEAKS_N1` | — | Mantener solo fits con este primer N de OGSE |
| `TC_PEAKS_N2` | — | Mantener solo fits con este segundo N de OGSE |
| `TC_PEAKS_EXTRA_ARGS` | — | Args extra para `plot_ogse-contrast_tc_peak_panels.py` |

```
nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse extract_tc_peak > logs/extract_tc_peak.log 2>&1 &

# Filtrar ROIs para los paneles
TC_PEAKS_ROIS="AntCC MidCC" \
  nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse extract_tc_peak > logs/extract_tc_peak.log 2>&1 &
```

---

### `tc` — ajustar tc vs td

Ajusta tc en función de td usando la tabla `tc_peak` producida por `extract_tc_peak`.

| Variable | Default | Descripción |
|---|---|---|
| `TC_FIT_PARAMS` | derivado automáticamente de `extract_tc_peak` | Parquet de tc_peak (paso 09) |
| `TC_METHOD` | `pseudohuber_fixed_macro` | `pseudohuber_free\|pseudohuber_fixed_macro\|linear` |
| `TC_Y_COL` | `tc_peak_ms` | `tc_peak_ms` o `tc_peak_resampled_ms` |
| `TC_OUT_DIR` | `$ANALYSIS_ROOT/tc/<master>` | Raíz de salida (cada método va a su subdir) |
| `TC_EXTRA_ARGS` | — | Args extra para `run_tc_vs_td.py` |

Cuando `TC_METHOD=pseudohuber_fixed_macro`, requiere pasar el excel de alpha:
```
TC_EXTRA_ARGS="--summary-alpha analysis/brains/ogse_experiments/alpha_macro/master/summary_alpha_values.xlsx"
```

```
# Método estándar (pseudohuber con alpha fijo desde alpha_macro)
TC_METHOD=pseudohuber_fixed_macro \
TC_EXTRA_ARGS="--summary-alpha analysis/brains/ogse_experiments/alpha_macro/master/summary_alpha_values.xlsx" \
  nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse tc > logs/tc.log 2>&1 &

# Método lineal con tc resampleado
TC_METHOD=linear TC_Y_COL=tc_peak_resampled_ms \
  nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse tc > logs/tc.log 2>&1 &

# Con parquet explícito
TC_FIT_PARAMS="analysis/brains/ogse_experiments/fits/master/ogse_value_norm_vs_bvaluethorsten_ogse_mixed_offset/tc_peak/tc_peak_table.parquet" \
TC_METHOD=pseudohuber_fixed_macro \
TC_EXTRA_ARGS="--summary-alpha analysis/brains/ogse_experiments/alpha_macro/master/summary_alpha_values.xlsx" \
  nohup bash nogse_pipeline/bash_template/run_dataset.sh brain ogse tc > logs/tc.log 2>&1 &
```