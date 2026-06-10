# Publication Figures

This repository keeps publication-specific figures separate from the core analysis pipeline. The current figure builders are:

- `scripts/plot_publication_delta_alpha.py`
- `scripts/plot_publication_contrast_lcf_tcpeak.py`
- `scripts/plot_publication_contrast_td_tcpeak.py`
- `scripts/plot_publication_tc_param_vars.py`

It reads already-generated OGSE result tables and writes editable PDF plus high-resolution PNG panels for:

- `delta` vs ROI,
- `alpha_macro` vs ROI,
- brain CC regions in `longitudinal` and `transversal`,
- phantom fibers in `longitudinal` and `transversal` where the default alias is `1=transversal`, `2=transversal`, and `3=longitudinal`; the two transversal directions are averaged.

The plotted brain values are individual subject parameter estimates with their own per-fit error bars. Brain subjects are displayed with generic labels (`subj1`, `subj2`, `subj3`) and separate marker shapes. Phantom values are collapsed to one value per fiber and direction, use a diamond marker, and do not show error bars because there are no subject-level replicates to average. Delta scales are independent for brain and phantom panels, while alpha uses a shared scale.

The contrast summary plots contrast vs corrected `lcf` for selected brain and phantom regions, with `tc_peak` vs `td` pseudo-Huber insets. Defaults use `CentralCC` for `BRAIN`, `fiber1` for `PHANTOM3`, brain directions `long/tra`, and phantom direction aliases `1=long`, `3=tra`. The phantom default excludes `td=209.1 ms`.

The selected-contrast panels reuse the same inputs and styling, but show one selected fitted/resampled signal and contrast panel on the left and the multi-`td` resampled contrast panel on the right, matched to the same direction. The right panel includes the same `tc_peak` vs `td` pseudo-Huber inset used by the `nogse_contrast_lcf_tcpeak` figures. Defaults use `PostCC`, direction `long`, `td=143.4 ms` for brains and `fiber1`, direction `1` shown as `longitudinal`, `td=142.5 ms` for phantoms.

The pseudo-Huber parameter panels adapt the `vars_sameY` and `delta_vs_alpha_macro` analysis plots for publication. They read the `tc_peak_resampled_data_ms` parameter tables, relabel brain subjects as `subj1`, `subj2`, and `subj3`, relabel directions as `longitudinal` and `transversal`, and use publication labels for `alpha_macro` and `delta`: `Macroscopic tortuosity coefficient \alpha` and `Transition parameter \delta [ms]`. For phantoms, the default transversal panel averages directions `1` and `2`; direction `3` is longitudinal.

## Default Run

```bash
bash bash_template/publication_figures/plot_delta_alpha_ogse.sh
bash bash_template/publication_figures/plot_nogse_contrast_lcf_tcpeak.sh
bash bash_template/publication_figures/plot_nogse_contrast_td_tcpeak.sh
bash bash_template/publication_figures/plot_tc_param_vars_pseudohuber.sh
```

Default outputs:

```text
analysis/publication_figures/delta_alpha_ogse/
analysis/publication_figures/nogse_contrast_lcf_tcpeak/
analysis/publication_figures/nogse_contrast_td_tcpeak/
analysis/publication_figures/tc_param_vars/
```

The script writes a combined brain+phantom figure, separate figures for each dataset, and CSV tables with the plotted values.

## Editing

Edit `bash_template/publication_figures/plot_delta_alpha_ogse.sh` to change:

- ROI order,
- directions,
- input alpha or delta tables,
- shared y-axis lower limits,
- output folder,
- PNG DPI,
- output formats.

Edit `bash_template/publication_figures/plot_nogse_contrast_lcf_tcpeak.sh` to change:

- brain/phantom subjects,
- ROI,
- direction aliases,
- contrast fit tables,
- pseudo-Huber parameter tables,
- excluded diffusion times,
- marker size and line width.

Edit `bash_template/publication_figures/plot_nogse_contrast_td_tcpeak.sh` to change:

- brain/phantom subjects,
- ROI,
- selected direction and selected `td`,
- contrast fit tables and resampled contrast roots,
- pseudo-Huber parameter tables,
- excluded diffusion times,
- marker size and line width.

Edit `bash_template/publication_figures/plot_tc_param_vars_pseudohuber.sh` to change:

- whether to build `vars`, `delta-alpha`, or both,
- plotted pseudo-Huber variables,
- ROI order,
- subject labels,
- direction labels and aliases,
- input `params_pseudohuber` tables,
- output folder,
- PNG DPI,
- output formats.

For deeper layout changes, edit `src/publication_figures/delta_alpha_roi.py`, `src/publication_figures/contrast_lcf_tcpeak.py`, or `src/publication_figures/tc_param_vars.py`. The PDF output keeps text as editable vector text where supported by the downstream editor.
