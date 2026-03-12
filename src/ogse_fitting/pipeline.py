from __future__ import annotations
import pandas as pd

from .config import OGSEFitConfig
from .data_loading import load_correction_table, load_all_curves
from tc_fittings.contrast_fit import fit_global
from .metrics import augment_params_and_plot_individual, compute_region_colors
from ..plottings.plots import (
    plot_individual_by_dirs,
    plot_globalfits_grid_G,
    plot_globalfits_grid_xvars,
    plot_param_comparisons,
)
from tc_fittings.tc_fits import run_posthoc

def run_all(cfg: OGSEFitConfig) -> pd.DataFrame:
    cfg.plot_dir_out.mkdir(parents=True, exist_ok=True)

    tabla_f = load_correction_table(cfg.correction_factors_xlsx)
    all_curves = load_all_curves(cfg, tabla_f)

    df_params = fit_global(cfg, all_curves, tabla_f)
    df_params = augment_params_and_plot_individual(cfg, all_curves, df_params, tabla_f)

    # Main plots
    plot_individual_by_dirs(cfg, all_curves, df_params, tabla_f)
    plot_globalfits_grid_G(cfg, all_curves, df_params, tabla_f)
    plot_globalfits_grid_xvars(cfg, all_curves, df_params, tabla_f)
    plot_param_comparisons(cfg, df_params, all_curves)

    # Post-hoc fits (tc(Td) lineal etc). Alpha_macro plots run only if summary exists.
    region2color = compute_region_colors(cfg, all_curves)
    run_posthoc(cfg, df_params, all_curves, region2color)

    return df_params
