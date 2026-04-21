from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd
from pathlib import Path

DEFAULT_PALETTE = ["#a65628", "#e41a1c", "#ff7f00", "#984ea3", "#377eb8", "#999999"]


@dataclass(frozen=True)
class FitSpec:
    name: str
    func: Callable
    default_k_last: Optional[int]
    needs_alpha_macro: bool
    description: str


from .tc_td_pseudohuber import (
    fit_tc_vs_td_pseudohuber,
    block1b_alpha_vs_Td,
    block1c_smallTd_tc_approx,
    block1d_fullrange_tc_with_approximations,
    block2_region_plots,
    block3_alpha_macro_summary_vs_fit,
    block4_qquad_vs_alpha_macro,
    block2b_cc_vars_long_tra_sameY,  # Now generic: uses all directions present.
)


def _ensure_direction_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the direction column name for the whole pipeline.
    Prefer 'direction', matching groupfits.
    """
    df = df.copy()
    if "direction" in df.columns:
        df["direction"] = df["direction"].astype(str).str.strip()
        return df
    if "direction" in df.columns:
        df = df.rename(columns={"direction": "direction"})
        df["direction"] = df["direction"].astype(str).str.strip()
        return df
    return df


def _regions_for_fit(df: pd.DataFrame, cfg) -> list[str]:
    if cfg is not None and getattr(cfg, "regions", None):
        return [str(r).replace("_norm", "") for r in cfg.regions]
    return sorted(df["roi"].astype(str).str.replace("_norm", "", regex=False).unique().tolist())


def _palette_for_fit(cfg) -> list[str]:
    if cfg is not None and getattr(cfg, "palette", None):
        return list(cfg.palette)
    return list(DEFAULT_PALETTE)


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
    # Fit
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

    # Normalize direction column for compatibility.
    df_fit = _ensure_direction_col(df_fit)
    df_params = _ensure_direction_col(df_params)

    # Blocks that do not depend on specific direction names.
    block1b_alpha_vs_Td(df_params, df_fit, out_dir, region_order=regions, td_min_ms=td_min_ms, td_max_ms=td_max_ms)
    block1c_smallTd_tc_approx(
        df_params,
        df_fit,
        out_dir,
        y_col=y_col,
        y_label=y_label,
        region_order=regions,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
    )
    block1d_fullrange_tc_with_approximations(
        df_params,
        df_fit,
        out_dir,
        y_col=y_col,
        y_label=y_label,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        region_order=regions,
    )
    block2_region_plots(df_fit, out_dir, regions, palette, plot_A=True, show_errorbars=show_errorbars)

    # Block2b is now generic: plot 1xN with all directions present.
    block2b_cc_vars_long_tra_sameY(
        df_fit=df_fit,
        out_dir=out_dir,
        cfg_regions=regions,
        palette=palette,
        show_errorbars=show_errorbars,
        tag=f"pseudohuber_free_k={k_last}_y={y_col}_mode={df_fit['mode'].unique()[0] if 'mode' in df_fit.columns else 'free_macro'}",
    )

    # Blocks 3/4 run only when an alpha_macro summary exists.
    if alpha_macro_df is not None:
        alpha_macro_df = _ensure_direction_col(alpha_macro_df)
        block3_alpha_macro_summary_vs_fit(
            df_fit, out_dir, alpha_macro_df, palette, method_tag=f"pseudohuber_free_k={k_last}", region_order=regions
        )
        block4_qquad_vs_alpha_macro(
            df_fit, out_dir, alpha_macro_df, palette, method_tag=f"pseudohuber_free_k={k_last}", region_order=regions
        )


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
    block1c_smallTd_tc_approx(
        df_params,
        df_fit,
        out_dir,
        y_col=y_col,
        y_label=y_label,
        region_order=regions,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
    )
    block1d_fullrange_tc_with_approximations(
        df_params,
        df_fit,
        out_dir,
        y_col=y_col,
        y_label=y_label,
        td_min_ms=td_min_ms,
        td_max_ms=td_max_ms,
        region_order=regions,
    )
    block2_region_plots(df_fit, out_dir, regions, palette, plot_A=True, show_errorbars=show_errorbars)

    # Generic: does not depend on long/tra labels.
    block2b_cc_vars_long_tra_sameY(
        df_fit=df_fit,
        out_dir=out_dir,
        cfg_regions=regions,
        palette=palette,
        show_errorbars=show_errorbars,
        tag=f"pseudohuber_fixed_macro_k={k_last}_y={y_col}_mode={df_fit['mode'].unique()[0] if 'mode' in df_fit.columns else 'fixed_macro'}",
    )

    if alpha_macro_df is not None:
        alpha_macro_df = _ensure_direction_col(alpha_macro_df)
        block3_alpha_macro_summary_vs_fit(
            df_fit, out_dir, alpha_macro_df, palette, method_tag=f"pseudohuber_fixed_macro_k={k_last}", region_order=regions
        )
        block4_qquad_vs_alpha_macro(
            df_fit, out_dir, alpha_macro_df, palette, method_tag=f"pseudohuber_fixed_macro_k={k_last}", region_order=regions
        )


METHODS = {
    "pseudohuber_free": FitSpec(
        name="pseudohuber_free",
        func=run_pseudohuber_free,
        default_k_last=None,
        needs_alpha_macro=False,
        description="PseudoHuber model: fit c, delta, and alpha_macro (free alpha_macro).",
    ),
    "pseudohuber_fixed_macro": FitSpec(
        name="pseudohuber_fixed_macro",
        func=run_pseudohuber_fixed_macro,
        default_k_last=None,
        needs_alpha_macro=True,
        description="PseudoHuber model: fix alpha_macro=alpha_macro(summary) and fit c, delta.",
    ),
}
