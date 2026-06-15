from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from tc_fittings.tc_td_models import tc_pseudohuber, tc_linear


DEFAULT_PALETTE = ["#a65628", "#e41a1c", "#ff7f00", "#984ea3", "#377eb8", "#999999"]


@dataclass(frozen=True)
class TcFitMethod:
    """
    Specification for a tc-vs-Td fitting model.

    To add a new model:
      1. Define the model function in tc_td_models.py:
             def tc_mymodel(Td, param1, param2): ...
      2. Add an entry to METHODS below:
             "mymodel": TcFitMethod(
                 name="mymodel",
                 model_func=tc_mymodel,
                 param_names=("param1", "param2"),
                 default_inits={"param1": 1.0, "param2": 0.1},
                 default_bounds={"param1": (0, np.inf), "param2": (0, np.inf)},
                 description="...",
             )
      3. Run it via scripts/fitting/run_tc_vs_td.py --method mymodel
    """
    name: str
    model_func: Callable
    param_names: tuple[str, ...]
    default_inits: dict[str, float]
    default_bounds: dict[str, tuple[float, float]]
    description: str
    needs_alpha_macro: bool = False


# ---------------------------------------------------------------------------
# Registered models
# ---------------------------------------------------------------------------

METHODS: dict[str, TcFitMethod] = {

    "pseudohuber_free": TcFitMethod(
        name="pseudohuber_free",
        model_func=tc_pseudohuber,
        param_names=("c", "delta", "alpha_macro"),
        default_inits={"c": 1.0, "delta": 100.0, "alpha_macro": 0.2},
        default_bounds={"c": (0.0, np.inf), "delta": (1e-6, np.inf), "alpha_macro": (0.1, 0.3)},
        description=(
            "Pseudo-Huber: tc(Td) = c + alpha*delta*(sqrt(1+(Td/delta)^2) - 1). "
            "Fits c, delta, alpha_macro freely."
        ),
        needs_alpha_macro=False,
    ),

    "pseudohuber_fixed_macro": TcFitMethod(
        name="pseudohuber_fixed_macro",
        model_func=tc_pseudohuber,
        param_names=("c", "delta", "alpha_macro"),
        default_inits={"c": 1.0, "delta": 100.0, "alpha_macro": 0.2},
        default_bounds={"c": (0.0, np.inf), "delta": (1e-6, np.inf), "alpha_macro": (0.1, 0.3)},
        description=(
            "Pseudo-Huber: tc(Td) = c + alpha*delta*(sqrt(1+(Td/delta)^2) - 1). "
            "alpha_macro is fixed from the summary table; fits only c and delta."
        ),
        needs_alpha_macro=True,
    ),

    "linear": TcFitMethod(
        name="linear",
        model_func=tc_linear,
        param_names=("c", "slope"),
        default_inits={"c": 1.0, "slope": 0.1},
        default_bounds={"c": (0.0, np.inf), "slope": (0.0, np.inf)},
        description="Linear: tc(Td) = c + slope * Td. Useful for the large-Td regime.",
        needs_alpha_macro=False,
    ),

}


# ---------------------------------------------------------------------------
# Run functions
#
# Each run_* function calls the appropriate fitting function and plot blocks.
# The pseudohuber models use fit_tc_vs_td_pseudohuber() because they have
# extra pseudohuber-specific features (fixed_macro mode, alpha_macro plots,
# q_quad plots, etc.).
#
# The linear model and any future simple models use the generic fit_tc_vs_td()
# from tc_td_pseudohuber — only a basic fit plot is produced.
# ---------------------------------------------------------------------------

from tc_fittings.tc_td_pseudohuber import (
    fit_tc_vs_td,
    fit_tc_vs_td_pseudohuber,
    block1b_alpha_vs_Td,
    block1c_smallTd_tc_approx,
    block1d_fullrange_tc_with_approximations,
    block2_region_plots,
    block2b_cc_vars_long_tra_sameY,
    block3_alpha_macro_summary_vs_fit,
    block4_qquad_vs_alpha_macro,
    block4_delta_vs_alpha_macro,
)


def _regions_for_fit(df: pd.DataFrame, cfg) -> list[str]:
    if cfg is not None and getattr(cfg, "regions", None):
        return [str(r).replace("_norm", "") for r in cfg.regions]
    return sorted(df["roi"].astype(str).str.replace("_norm", "", regex=False).unique().tolist())


def _palette_for_fit(cfg) -> list[str]:
    if cfg is not None and getattr(cfg, "palette", None):
        return list(cfg.palette)
    return list(DEFAULT_PALETTE)


def _ensure_direction_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "direction" in df.columns:
        df["direction"] = df["direction"].astype(str).str.strip()
    return df


def run_pseudohuber_free(
    cfg,
    df_params: pd.DataFrame,
    out_dir: Path,
    k_last: Optional[int],
    alpha_macro_df,
    *,
    y_col: str = "tc_peak_ms",
    y_label: str = r"$t_{c,peak}$ [ms]",
    show_errorbars: bool = True,
    td_min_ms: float = 0.0,
    td_max_ms: float = 2000.0,
    c_fixed: float | None = None,
    c_min: float = 0.0,
    c_max: float = float("inf"),
    delta_fixed: float | None = None,
    delta_min: float = 1e-6,
    delta_max: float = float("inf"),
    alpha_macro_fixed: float | None = None,
    alpha_macro_min: float = 0.1,
    alpha_macro_max: float = 0.3,
):
    regions = _regions_for_fit(df_params, cfg)
    palette = _palette_for_fit(cfg)
    df_fit = fit_tc_vs_td_pseudohuber(
        df_params=df_params,
        out_dir=out_dir,
        cfg_regions=regions,
        palette=palette,
        k_last=k_last,
        mode="free_macro",
        alpha_macro_df=None,
        y_col=y_col,
        y_label=y_label,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        c_fixed=c_fixed,
        c_min=c_min,
        c_max=c_max,
        delta_fixed=delta_fixed,
        delta_min=delta_min,
        delta_max=delta_max,
        alpha_macro_fixed=alpha_macro_fixed,
        alpha_macro_min=alpha_macro_min,
        alpha_macro_max=alpha_macro_max,
    )
    df_fit = _ensure_direction_col(df_fit)
    df_params = _ensure_direction_col(df_params)

    block1b_alpha_vs_Td(df_params, df_fit, out_dir, region_order=regions, td_min_ms=td_min_ms, td_max_ms=td_max_ms)
    block1c_smallTd_tc_approx(df_params, df_fit, out_dir, y_col=y_col, y_label=y_label, region_order=regions, td_min_ms=td_min_ms, td_max_ms=td_max_ms)
    block1d_fullrange_tc_with_approximations(df_params, df_fit, out_dir, y_col=y_col, y_label=y_label, td_min_ms=td_min_ms, td_max_ms=td_max_ms, region_order=regions)
    block2_region_plots(df_fit, out_dir, regions, palette, plot_A=True, show_errorbars=show_errorbars)
    block2b_cc_vars_long_tra_sameY(df_fit=df_fit, out_dir=out_dir, cfg_regions=regions, palette=palette, show_errorbars=show_errorbars, tag=f"pseudohuber_free_k={k_last}_y={y_col}")

    if alpha_macro_df is not None:
        alpha_macro_df = _ensure_direction_col(alpha_macro_df)
        block3_alpha_macro_summary_vs_fit(df_fit, out_dir, alpha_macro_df, palette, method_tag=f"pseudohuber_free_k={k_last}", region_order=regions)
        block4_qquad_vs_alpha_macro(df_fit, out_dir, alpha_macro_df, palette, method_tag=f"pseudohuber_free_k={k_last}", region_order=regions)
        block4_delta_vs_alpha_macro(df_fit, out_dir, alpha_macro_df, palette, method_tag=f"pseudohuber_free_k={k_last}", region_order=regions)


def run_pseudohuber_fixed_macro(
    cfg,
    df_params: pd.DataFrame,
    out_dir: Path,
    k_last: Optional[int],
    alpha_macro_df,
    *,
    y_col: str = "tc_peak_ms",
    y_label: str = r"$t_{c,peak}$ [ms]",
    show_errorbars: bool = True,
    td_min_ms: float = 0.0,
    td_max_ms: float = 2000.0,
    c_fixed: float | None = None,
    c_min: float = 0.0,
    c_max: float = float("inf"),
    delta_fixed: float | None = None,
    delta_min: float = 1e-6,
    delta_max: float = float("inf"),
    alpha_macro_fixed: float | None = None,
    alpha_macro_min: float = 0.1,
    alpha_macro_max: float = 0.3,
):
    regions = _regions_for_fit(df_params, cfg)
    palette = _palette_for_fit(cfg)
    df_fit = fit_tc_vs_td_pseudohuber(
        df_params=df_params,
        out_dir=out_dir,
        cfg_regions=regions,
        palette=palette,
        k_last=k_last,
        mode="fixed_macro",
        alpha_macro_df=alpha_macro_df,
        y_col=y_col,
        y_label=y_label,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        c_fixed=c_fixed,
        c_min=c_min,
        c_max=c_max,
        delta_fixed=delta_fixed,
        delta_min=delta_min,
        delta_max=delta_max,
        alpha_macro_fixed=alpha_macro_fixed,
        alpha_macro_min=alpha_macro_min,
        alpha_macro_max=alpha_macro_max,
    )
    df_fit = _ensure_direction_col(df_fit)
    df_params = _ensure_direction_col(df_params)

    block1b_alpha_vs_Td(df_params, df_fit, out_dir, region_order=regions, td_min_ms=td_min_ms, td_max_ms=td_max_ms)
    block1c_smallTd_tc_approx(df_params, df_fit, out_dir, y_col=y_col, y_label=y_label, region_order=regions, td_min_ms=td_min_ms, td_max_ms=td_max_ms)
    block1d_fullrange_tc_with_approximations(df_params, df_fit, out_dir, y_col=y_col, y_label=y_label, td_min_ms=td_min_ms, td_max_ms=td_max_ms, region_order=regions)
    block2_region_plots(df_fit, out_dir, regions, palette, plot_A=True, show_errorbars=show_errorbars)
    block2b_cc_vars_long_tra_sameY(df_fit=df_fit, out_dir=out_dir, cfg_regions=regions, palette=palette, show_errorbars=show_errorbars, tag=f"pseudohuber_fixed_macro_k={k_last}_y={y_col}")

    if alpha_macro_df is not None:
        alpha_macro_df = _ensure_direction_col(alpha_macro_df)
        block3_alpha_macro_summary_vs_fit(df_fit, out_dir, alpha_macro_df, palette, method_tag=f"pseudohuber_fixed_macro_k={k_last}", region_order=regions)
        block4_qquad_vs_alpha_macro(df_fit, out_dir, alpha_macro_df, palette, method_tag=f"pseudohuber_fixed_macro_k={k_last}", region_order=regions)
        block4_delta_vs_alpha_macro(df_fit, out_dir, alpha_macro_df, palette, method_tag=f"pseudohuber_fixed_macro_k={k_last}", region_order=regions)


def run_linear(
    cfg,
    df_params: pd.DataFrame,
    out_dir: Path,
    k_last: Optional[int],
    alpha_macro_df,  # unused for linear, kept for uniform call signature
    *,
    y_col: str = "tc_peak_ms",
    y_label: str = r"$t_{c,peak}$ [ms]",
    show_errorbars: bool = True,
    td_min_ms: float = 0.0,
    td_max_ms: float = 2000.0,
    c_min: float = 0.0,
    c_max: float = float("inf"),
    slope_min: float = 0.0,
    slope_max: float = float("inf"),
    **_ignored,  # absorb pseudohuber-specific kwargs (delta_*, alpha_macro_*) from the script
):
    """Run a simple linear tc(Td) = c + slope * Td fit."""
    regions = _regions_for_fit(df_params, cfg)
    palette = _palette_for_fit(cfg)
    method = METHODS["linear"]
    df_fit = fit_tc_vs_td(
        df_params,
        model_func=method.model_func,
        param_names=method.param_names,
        param_inits=method.default_inits,
        param_bounds={
            "c": (c_min, c_max),
            "slope": (slope_min, slope_max),
        },
        k_last=k_last,
        y_col=y_col,
        y_label=y_label,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        out_dir=out_dir,
        cfg_regions=regions,
        palette=palette,
    )
    df_fit = _ensure_direction_col(df_fit)
    # block2b works for any model: it plots fit params per direction
    block2b_cc_vars_long_tra_sameY(
        df_fit=df_fit,
        out_dir=out_dir,
        cfg_regions=regions,
        palette=palette,
        show_errorbars=show_errorbars,
        tag=f"linear_k={k_last}_y={y_col}",
    )


# ---------------------------------------------------------------------------
# Map method name → run function (used by run_tc_vs_td.py script)
# ---------------------------------------------------------------------------

_RUN_FUNCTIONS: dict[str, Callable] = {
    "pseudohuber_free": run_pseudohuber_free,
    "pseudohuber_fixed_macro": run_pseudohuber_fixed_macro,
    "linear": run_linear,
}


def get_run_function(method_name: str) -> Callable:
    """Return the run function for a registered method name."""
    if method_name not in _RUN_FUNCTIONS:
        available = sorted(_RUN_FUNCTIONS)
        raise KeyError(f"Unknown tc fit method {method_name!r}. Available: {available}")
    return _RUN_FUNCTIONS[method_name]
