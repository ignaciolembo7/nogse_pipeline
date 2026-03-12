from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd
from pathlib import Path

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
    block2_region_plots,
    block3_alpha_macro_vs_micro,
    block4_qquad_vs_alpha_macro,
    block2b_cc_vars_long_tra_sameY,  # ahora es genérico (usa todas las direcciones presentes)
)


def _ensure_direction_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza el nombre de la columna de direction para todo el pipeline.
    Preferimos 'direction' (como el globalfit).
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


def run_pseudohuber_free(cfg, df_params: pd.DataFrame, out_dir: Path, k_last: Optional[int], alpha_macro_df):
    # Fit
    df_fit = fit_tc_vs_td_pseudohuber(
        df_params=df_params,
        out_dir=out_dir,
        cfg_regions=cfg.regions,
        palette=cfg.palette,
        k_last=k_last,
        mode="free_alpha",
        alpha_macro_df=None,
    )

    # Normalizar col de direction (por compatibilidad)
    df_fit = _ensure_direction_col(df_fit)
    df_params = _ensure_direction_col(df_params)

    # Blocks que no dependen de nombres específicos de direction
    block1b_alpha_vs_Td(df_params, df_fit, out_dir)
    block1c_smallTd_tc_approx(df_params, df_fit, out_dir)
    block2_region_plots(df_fit, out_dir, cfg.regions, cfg.palette, plot_A=True)

    # ✅ Block2b ahora es genérico: plotea 1×N con TODAS las direcciones presentes
    block2b_cc_vars_long_tra_sameY(
        df_fit=df_fit,
        out_dir=out_dir,
        cfg_regions=cfg.regions,
        palette=cfg.palette,
        tag=f"pseudohuber_free_k={k_last}_mode={df_fit['mode'].unique()[0] if 'mode' in df_fit.columns else 'free_alpha'}",
    )

    # Block3/4 solo si hay summary alpha_macro (macro)
    if alpha_macro_df is not None:
        alpha_macro_df = _ensure_direction_col(alpha_macro_df)
        block3_alpha_macro_vs_micro(
            df_fit, out_dir, alpha_macro_df, cfg.palette, method_tag=f"pseudohuber_free_k={k_last}"
        )
        block4_qquad_vs_alpha_macro(
            df_fit, out_dir, alpha_macro_df, cfg.palette, method_tag=f"pseudohuber_free_k={k_last}"
        )


def run_pseudohuber_fixed_macro(cfg, df_params: pd.DataFrame, out_dir: Path, k_last: Optional[int], alpha_macro_df):
    df_fit = fit_tc_vs_td_pseudohuber(
        df_params=df_params,
        out_dir=out_dir,
        cfg_regions=cfg.regions,
        palette=cfg.palette,
        k_last=k_last,
        mode="fixed_macro",
        alpha_macro_df=alpha_macro_df,
    )

    df_fit = _ensure_direction_col(df_fit)
    df_params = _ensure_direction_col(df_params)

    block1b_alpha_vs_Td(df_params, df_fit, out_dir)
    block1c_smallTd_tc_approx(df_params, df_fit, out_dir)
    block2_region_plots(df_fit, out_dir, cfg.regions, cfg.palette, plot_A=True)

    # ✅ Genérico (no depende de long/tra)
    block2b_cc_vars_long_tra_sameY(
        df_fit=df_fit,
        out_dir=out_dir,
        cfg_regions=cfg.regions,
        palette=cfg.palette,
        tag=f"pseudohuber_fixed_macro_k={k_last}_mode={df_fit['mode'].unique()[0] if 'mode' in df_fit.columns else 'fixed_macro'}",
    )

    if alpha_macro_df is not None:
        alpha_macro_df = _ensure_direction_col(alpha_macro_df)
        block3_alpha_macro_vs_micro(
            df_fit, out_dir, alpha_macro_df, cfg.palette, method_tag=f"pseudohuber_fixed_macro_k={k_last}"
        )
        block4_qquad_vs_alpha_macro(
            df_fit, out_dir, alpha_macro_df, cfg.palette, method_tag=f"pseudohuber_fixed_macro_k={k_last}"
        )


METHODS = {
    "pseudohuber_free": FitSpec(
        name="pseudohuber_free",
        func=run_pseudohuber_free,
        default_k_last=None,
        needs_alpha_macro=False,
        description="PseudoHuber model: ajusta c, delta, alpha_inf (alpha_inf libre).",
    ),
    "pseudohuber_fixed_macro": FitSpec(
        name="pseudohuber_fixed_macro",
        func=run_pseudohuber_fixed_macro,
        default_k_last=None,
        needs_alpha_macro=True,
        description="PseudoHuber model: fija alpha_inf=alpha_macro(summary) y ajusta c, delta.",
    ),
}