from __future__ import annotations

from pathlib import Path
from typing import Sequence
import warnings

import matplotlib.cm as cm
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ogse_fitting.contrast_fit_panels import (
    _build_fit_curve,
    _extract_plot_arrays,
    _load_contrast_table_cached,
    _panel_specs,
    _pick_order,
    _resolve_contrast_parquet_for_row,
    _sanitize_token,
    _subset_group,
)
from ogse_fitting.fit_ogse_contrast_vs_g import _model_yhat
from tc_fittings.contrast_fit_table import load_contrast_fit_params


VALID_X_VARS = ("g", "Ld", "lcf", "lcf_a", "tc")
X_VAR_ALIASES = {"Lcf": "lcf_a"}
VALID_PEAK_SOURCES = ("standard", "resampled", "both")
VALID_CONTRAST_SOURCES = ("direct", "fitted_resampled", "auto")
RESAMPLED_DATA_PEAK_TC_COL = "tc_peak_resampled_data_ms"
RESAMPLED_DATA_PEAK_SIGNAL_COL = "signal_peak_resampled_data"
RESAMPLED_DATA_PEAK_G_COL = "g_peak_resampled_data_corr_mTm"


def _derived_axes_from_g(
    g_mTm: np.ndarray,
    *,
    td_ms: float,
    peak_D0_fix: float,
    peak_gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = np.asarray(g_mTm, dtype=float)
    Ld = np.full_like(g, np.nan, dtype=float)
    lcf_um = np.full_like(g, np.nan, dtype=float)
    lcf_a = np.full_like(g, np.nan, dtype=float)
    tc_ms = np.full_like(g, np.nan, dtype=float)

    valid = np.isfinite(g) & (g > 0)
    if not np.any(valid):
        return Ld, lcf_um, lcf_a, tc_ms

    D0 = float(peak_D0_fix)
    gamma = float(peak_gamma)
    td = float(td_ms)
    l_d = np.sqrt(D0 * td)
    l_G = (D0 / (gamma * g[valid])) ** (1.0 / 3.0)
    Ld_valid = l_d / l_G
    Lcf_valid = ((3.0 / 2.0) ** (1.0 / 4.0)) * (Ld_valid ** (-1.0 / 2.0))
    lcf_valid_um = (Lcf_valid * l_G) * 1e6
    tc_valid_ms = ((lcf_valid_um * 1e-6) ** 2) / D0

    Ld[valid] = Ld_valid
    lcf_a[valid] = Lcf_valid
    lcf_um[valid] = lcf_valid_um
    tc_ms[valid] = tc_valid_ms
    return Ld, lcf_um, lcf_a, tc_ms


def _transform_x(
    *,
    xvar: str,
    active_corr: np.ndarray,
    fit_row: pd.Series,
    peak_D0_fix: float,
    peak_gamma: float,
) -> np.ndarray:
    xvar = str(xvar)
    if xvar == "g":
        return np.asarray(active_corr, dtype=float)

    Ld, lcf_um, lcf_a, tc_ms = _derived_axes_from_g(
        active_corr,
        td_ms=float(fit_row["td_ms"]),
        peak_D0_fix=peak_D0_fix,
        peak_gamma=peak_gamma,
    )
    if xvar == "Ld":
        return Ld
    if xvar == "lcf":
        return lcf_um
    if xvar == "lcf_a":
        return lcf_a
    if xvar == "tc":
        return tc_ms
    raise ValueError(f"Unsupported xvar: {xvar!r}. Expected one of {VALID_X_VARS}.")


def _peak_point_for_xvar(
    fit_row: pd.Series,
    *,
    xvar: str,
    peak_D0_fix: float,
    peak_gamma: float,
    peak_source: str = "standard",
) -> tuple[float, float]:
    source = str(peak_source)
    if source == "resampled":
        g_col = "g_peak_resampled_corr_mTm"
        tc_col = "tc_peak_resampled_ms"
        g_corr = float(pd.to_numeric(pd.Series([fit_row.get(g_col, np.nan)]), errors="coerce").iloc[0])
        if not np.isfinite(g_corr) or g_corr <= 0:
            return np.nan, np.nan
        if str(fit_row.get("method", "")).startswith("signal_fit_1_minus_signal_fit_2"):
            y_peak = float(pd.to_numeric(pd.Series([fit_row.get("signal_peak", np.nan)]), errors="coerce").iloc[0])
        else:
            y_curve = _model_yhat(
                model=str(fit_row["model"]),
                td_ms=float(fit_row["td_ms"]),
                G1=np.asarray([g_corr], dtype=float),
                G2=np.asarray([g_corr], dtype=float),
                n_1=int(fit_row["N_1"]),
                n_2=int(fit_row["N_2"]),
                fit_row=fit_row.to_dict(),
            )
            y_peak = float(np.asarray(y_curve, dtype=float)[0])
    else:
        g_col = "x_peak_corr_mTm"
        tc_col = "tc_peak_ms"
        y_peak = float(pd.to_numeric(pd.Series([fit_row.get("signal_peak", np.nan)]), errors="coerce").iloc[0])

    if not np.isfinite(y_peak):
        return np.nan, np.nan

    if xvar == "g":
        x_peak = float(pd.to_numeric(pd.Series([fit_row.get(g_col, np.nan)]), errors="coerce").iloc[0])
        return x_peak, y_peak

    g_corr = float(pd.to_numeric(pd.Series([fit_row.get(g_col, np.nan)]), errors="coerce").iloc[0])
    Ld, lcf_um, lcf_a, tc_ms = _derived_axes_from_g(
        np.asarray([g_corr], dtype=float),
        td_ms=float(fit_row["td_ms"]),
        peak_D0_fix=peak_D0_fix,
        peak_gamma=peak_gamma,
    )
    if xvar == "Ld":
        return float(Ld[0]), y_peak
    if xvar == "lcf":
        return float(lcf_um[0]), y_peak
    if xvar == "lcf_a":
        return float(lcf_a[0]), y_peak
    if xvar == "tc":
        tc_peak = float(pd.to_numeric(pd.Series([fit_row.get(tc_col, np.nan)]), errors="coerce").iloc[0])
        return float(tc_peak if np.isfinite(tc_peak) else tc_ms[0]), y_peak
    raise ValueError(f"xvar no soportada: {xvar!r}")


def _peak_point_from_curve(
    *,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[float, float]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return np.nan, np.nan
    xm = x[m]
    ym = y[m]
    idx = int(np.nanargmax(ym))
    return float(xm[idx]), float(ym[idx])


def _build_resampled_fit_curve(
    fit_row: pd.Series,
    *,
    n_points: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    resampled_max = float(pd.to_numeric(pd.Series([fit_row.get("peak_resample_g_max_corr_mTm", np.nan)]), errors="coerce").iloc[0])
    if np.isfinite(resampled_max) and resampled_max > 0:
        G = np.linspace(0.0, resampled_max, max(32, int(n_points)))
        y_curve = _model_yhat(
            model=str(fit_row["model"]),
            td_ms=float(fit_row["td_ms"]),
            G1=G,
            G2=G,
            n_1=int(fit_row["N_1"]),
            n_2=int(fit_row["N_2"]),
            fit_row=fit_row.to_dict(),
        )
        return G, np.asarray(y_curve, dtype=float)

    maxima = pd.to_numeric(
        pd.Series(
            [
                fit_row.get("g1_search_max_corr_mTm", np.nan),
                fit_row.get("g2_search_max_corr_mTm", np.nan),
                fit_row.get("g1_max_corr_mTm", np.nan),
                fit_row.get("g2_max_corr_mTm", np.nan),
            ]
        ),
        errors="coerce",
    ).to_numpy(dtype=float)
    maxima = maxima[np.isfinite(maxima) & (maxima > 0)]
    if maxima.size == 0:
        return np.array([]), np.array([])

    G = np.linspace(0.0, float(np.nanmax(maxima)), max(32, int(n_points)))
    y_curve = _model_yhat(
        model=str(fit_row["model"]),
        td_ms=float(fit_row["td_ms"]),
        G1=G,
        G2=G,
        n_1=int(fit_row["N_1"]),
        n_2=int(fit_row["N_2"]),
        fit_row=fit_row.to_dict(),
    )
    return G, np.asarray(y_curve, dtype=float)


def add_resampled_data_peak_columns(
    df: pd.DataFrame,
    *,
    contrast_root: str | Path,
    peak_D0_fix: float = 3.2e-12,
    peak_gamma: float = 267.5221900,
    warn: bool = True,
) -> pd.DataFrame:
    """Add peak columns matching the resampled tc-panel data markers.

    These peaks are taken directly from the plotted resampled contrast table:
    convert the resampled gradient axis to tc, then keep the tc value at the
    maximum finite contrast point. This is intentionally distinct from
    tc_peak_resampled_ms, which is computed from the fitted contrast model.
    """
    out = df.copy()
    for col in (RESAMPLED_DATA_PEAK_TC_COL, RESAMPLED_DATA_PEAK_SIGNAL_COL, RESAMPLED_DATA_PEAK_G_COL):
        if col not in out.columns:
            out[col] = np.nan

    cache: dict[Path, pd.DataFrame] = {}
    missing_items: list[str] = []

    for idx, row in out.iterrows():
        try:
            contrast_path = _resolve_contrast_parquet_for_row(
                row,
                contrast_root=contrast_root,
                cache=cache,
            )
            contrast_df = _load_contrast_table_cached(contrast_path, cache)
            df_group = _subset_group(contrast_df, row)
            if df_group.empty:
                missing_items.append(f"{row.get('analysis_id', idx)} | {row.get('roi', '')} | {row.get('direction', '')} -> empty group")
                continue

            x_corr, y, _, _ = _extract_plot_arrays(df_group, row)
            tc_values = _transform_x(
                xvar="tc",
                active_corr=x_corr,
                fit_row=row,
                peak_D0_fix=peak_D0_fix,
                peak_gamma=peak_gamma,
            )
            m = np.isfinite(tc_values) & np.isfinite(y) & np.isfinite(x_corr)
            if not np.any(m):
                missing_items.append(f"{row.get('analysis_id', idx)} | {row.get('roi', '')} | {row.get('direction', '')} -> no finite tc peak")
                continue

            x_m = x_corr[m]
            tc_m = tc_values[m]
            y_m = y[m]
            i_peak = int(np.nanargmax(y_m))
            out.at[idx, RESAMPLED_DATA_PEAK_TC_COL] = float(tc_m[i_peak])
            out.at[idx, RESAMPLED_DATA_PEAK_SIGNAL_COL] = float(y_m[i_peak])
            out.at[idx, RESAMPLED_DATA_PEAK_G_COL] = float(x_m[i_peak])
        except Exception as exc:
            missing_items.append(f"{row.get('analysis_id', idx)} | {row.get('roi', '')} | {row.get('direction', '')} -> {exc}")

    if warn and missing_items:
        warnings.warn(
            "Some resampled data peaks could not be computed:\n" + "\n".join(missing_items[:15]),
            stacklevel=2,
        )

    return out


def _x_label(spec_gbase: str, spec_xplot: str, xvar: str, *, use_resampled_gradient: bool = False) -> str:
    if xvar == "g":
        if use_resampled_gradient:
            return "g_resampled [mT/m]"
        return f"{spec_gbase}_{spec_xplot} [mT/m]"
    if xvar == "Ld":
        return r"$L_d$"
    if xvar == "lcf":
        return r"$l_{cf}$ [$\mu$m]"
    if xvar == "lcf_a":
        return r"$l_{cf,a}$"
    if xvar == "tc":
        return r"$t_c$ [ms]"
    return xvar


def _gradient_output_token(spec_xplot: str, *, use_resampled_gradient: bool) -> str:
    if use_resampled_gradient:
        return "gradient=g_resampled"
    return f"xplot={_sanitize_token(spec_xplot)}"


def _validate_x_vars(x_vars: Sequence[str]) -> list[str]:
    out = [X_VAR_ALIASES.get(str(x), str(x)) for x in x_vars]
    invalid = [x for x in out if x not in VALID_X_VARS]
    if invalid:
        raise ValueError(f"Invalid x_vars: {invalid}. Expected one of {VALID_X_VARS}.")
    return list(dict.fromkeys(out))


def _validate_peak_marker_x_vars(x_vars: Sequence[str] | None) -> set[str]:
    if x_vars is None:
        return {"tc"}
    out = [X_VAR_ALIASES.get(str(x), str(x)) for x in x_vars]
    if any(x.upper() == "NONE" for x in out):
        return set()
    if any(x.upper() == "ALL" for x in out):
        return set(VALID_X_VARS)
    invalid = [x for x in out if x not in VALID_X_VARS]
    if invalid:
        raise ValueError(f"Invalid peak_marker_x_vars: {invalid}. Expected one of {VALID_X_VARS}, ALL, or NONE.")
    return set(dict.fromkeys(out))


def _normalize_x_lims(x_lims: dict[str, tuple[float, float]] | None) -> dict[str, tuple[float, float]]:
    if not x_lims:
        return {}

    out: dict[str, tuple[float, float]] = {}
    for raw_name, raw_lims in x_lims.items():
        name = X_VAR_ALIASES.get(str(raw_name), str(raw_name))
        if name not in VALID_X_VARS:
            raise ValueError(f"x_lims contains invalid xvar: {raw_name!r}. Expected one of {VALID_X_VARS}.")
        xmin, xmax = map(float, raw_lims)
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            raise ValueError(f"Invalid x_lims for {raw_name!r}: {(xmin, xmax)}")
        out[name] = (xmin, xmax)
    return out


def plot_contrast_tc_peak_panels(
    *,
    fits_root: str | Path,
    contrast_root: str | Path,
    out_dir: str | Path,
    pattern: str = "**/fit_params.*",
    models: list[str] | None = None,
    subjs: list[str] | None = None,
    rois: list[str] | None = None,
    directions: list[str] | None = None,
    exclude_td_ms: list[float] | None = None,
    x_vars: Sequence[str] = ("g", "Ld", "lcf", "lcf_a", "tc"),
    peak_marker_x_vars: Sequence[str] | None = ("tc",),
    peak_source: str = "standard",
    show_resampled_fit: bool = False,
    show_data_points: bool = True,
    resampled_curve_n: int = 300,
    peak_D0_fix: float = 3.2e-12,
    peak_gamma: float = 267.5221900,
    x_lims: dict[str, tuple[float, float]] | None = None,
    contrast_source: str = "auto",
    n1: int | None = None,
    n2: int | None = None,
    exclude_fitresamp: bool = False,
    ok_only: bool = True,
) -> list[Path]:
    x_vars = _validate_x_vars(x_vars)
    peak_marker_x_vars = _validate_peak_marker_x_vars(peak_marker_x_vars)
    peak_source = str(peak_source)
    if peak_source not in VALID_PEAK_SOURCES:
        raise ValueError(f"Invalid peak_source={peak_source!r}. Expected one of {VALID_PEAK_SOURCES}.")
    contrast_source = str(contrast_source)
    if contrast_source not in VALID_CONTRAST_SOURCES:
        raise ValueError(f"Invalid contrast_source={contrast_source!r}. Expected one of {VALID_CONTRAST_SOURCES}.")
    if contrast_source == "auto":
        contrast_source = "fitted_resampled" if (Path(contrast_root).name == "contrast") else "direct"
    x_lims = _normalize_x_lims(x_lims)
    use_resampled_gradient = bool(contrast_source == "fitted_resampled" or (show_resampled_fit and not show_data_points))
    print(f"[diagnostic] fits_root={Path(fits_root)}")
    print(f"[diagnostic] contrast_root={Path(contrast_root)}")
    print(f"[diagnostic] fit_params pattern={pattern}")
    print(f"[diagnostic] contrast_source={contrast_source}")
    print(f"[diagnostic] requested N pair={n1 if n1 is not None else 'ALL'}-{n2 if n2 is not None else 'ALL'}")
    raw_df = load_contrast_fit_params(
        [fits_root],
        pattern=pattern,
        models=None,
        subjs=None,
        directions=None,
        rois=None,
        exclude_fitresamp=False,
        ok_only=False,
    )
    print(f"[diagnostic] fit rows before filters={len(raw_df)}")
    df = load_contrast_fit_params(
        [fits_root],
        pattern=pattern,
        models=models,
        subjs=subjs,
        directions=directions,
        rois=rois,
        exclude_fitresamp=bool(exclude_fitresamp),
        ok_only=ok_only,
    )

    if df.empty:
        raise ValueError("No valid fits remained after filtering.")

    if n1 is not None and "N_1" in df.columns:
        df = df[pd.to_numeric(df["N_1"], errors="coerce") == int(n1)].copy()

    if n2 is not None and "N_2" in df.columns:
        df = df[pd.to_numeric(df["N_2"], errors="coerce") == int(n2)].copy()

    if df.empty:
        raise ValueError("No valid fits remained after filtering N_1/N_2.")

    if exclude_td_ms:
        td_vals = pd.to_numeric(df["td_ms"], errors="coerce")
        keep = np.ones(len(df), dtype=bool)
        for td_excl in exclude_td_ms:
            keep &= ~np.isclose(td_vals.to_numpy(dtype=float), float(td_excl), atol=1e-3, equal_nan=False)
        df = df.loc[keep].copy()
        if df.empty:
            raise ValueError("No valid fits remained after excluding td_ms.")

    dedup_cols = [
        col
        for col in (
            "analysis_id",
            "sheet",
            "subj",
            "roi",
            "direction",
            "stat",
            "td_ms",
            "N_1",
            "N_2",
            "model",
            "gbase",
            "ycol",
            "xplot",
        )
        if col in df.columns
    ]
    before_dedup = len(df)
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols, keep="last").copy()

    print(f"[diagnostic] fit rows after filters={len(df)}")
    if before_dedup != len(df):
        print(f"[diagnostic] duplicate fit rows removed={before_dedup - len(df)}")
    n_cols = [col for col in ("N_1", "N_2", "N1", "N2", "n_1", "n_2", "n1", "n2") if col in df.columns]
    print(f"[diagnostic] detected N columns={n_cols}")
    if {"N_1", "N_2"}.issubset(df.columns):
        n_pairs = df[["N_1", "N_2"]].drop_duplicates().sort_values(["N_1", "N_2"]).to_dict("records")
        print(f"[diagnostic] unique N pairs={n_pairs}")
    if "td_ms" in df.columns:
        td_detected = sorted(pd.to_numeric(df["td_ms"], errors="coerce").dropna().unique().tolist())
        print(f"[diagnostic] detected td_ms values={td_detected}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict[Path, pd.DataFrame] = {}
    outputs: list[Path] = []

    for spec in _panel_specs(df):
        sub = df[
            (df["subj"].astype(str) == spec.subj)
            & (df["model"].astype(str) == spec.model)
            & (df["gbase"].astype(str) == spec.gbase)
            & (df["ycol"].astype(str) == spec.ycol)
            & (df["xplot"].astype(str) == spec.xplot)
        ].copy()
        if sub.empty:
            continue

        roi_order = _pick_order(rois, sub["roi"].astype(str).unique().tolist())
        dir_order = _pick_order(directions, sub["direction"].astype(str).unique().tolist(), preferred=("long", "tra"))
        if not roi_order or not dir_order:
            continue

        td_values = sorted(pd.to_numeric(sub["td_ms"], errors="coerce").dropna().unique().tolist())
        if not td_values:
            continue

        cmap = cm.get_cmap("viridis", max(2, len(td_values)))
        td_to_color = {float(td): cmap(i) for i, td in enumerate(td_values)}

        for xvar in x_vars:
            fig, axes = plt.subplots(
                len(roi_order),
                len(dir_order),
                figsize=(6.4 * len(dir_order), 3.8 * len(roi_order) + 1.1),
                sharey=True,
                squeeze=False,
            )

            missing_items: list[str] = []

            for i, roi in enumerate(roi_order):
                for j, direction in enumerate(dir_order):
                    ax = axes[i, j]
                    sub_panel = sub[(sub["roi"].astype(str) == roi) & (sub["direction"].astype(str) == direction)].copy()
                    sub_panel = sub_panel.sort_values("td_ms", kind="stable")
                    x_bounds: list[np.ndarray] = []

                    if sub_panel.empty:
                        ax.set_title(f"{roi} | {direction}")
                        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                        ax.grid(True, alpha=0.25)
                        continue

                    for _, row in sub_panel.iterrows():
                        td_val = float(row["td_ms"])
                        color = td_to_color.get(td_val, "#1f77b4")
                        try:
                            curve_x_plot = np.array([])
                            curve_y_plot = np.array([])

                            if contrast_source == "fitted_resampled":
                                contrast_path = _resolve_contrast_parquet_for_row(
                                    row,
                                    contrast_root=contrast_root,
                                    cache=cache,
                                )
                                print(
                                    "[diagnostic] loading contrast table "
                                    f"analysis_id={row['analysis_id']} sheet={row.get('sheet', None)} "
                                    f"roi={roi} direction={direction} td_ms={row.get('td_ms', None)} "
                                    f"N=({row.get('N_1', None)}, {row.get('N_2', None)}) path={contrast_path}"
                                )
                                contrast_df = _load_contrast_table_cached(contrast_path, cache)
                                before_group = len(contrast_df)
                                df_group = _subset_group(contrast_df, row)
                                after_group = len(df_group)
                                print(
                                    "[diagnostic] contrast points "
                                    f"before_filter={before_group} after_filter={after_group}"
                                )
                                if df_group.empty:
                                    missing_items.append(f"{row['analysis_id']} | {roi} | {direction} -> empty group")
                                    continue

                                x_corr, y, _, _ = _extract_plot_arrays(df_group, row)
                                if x_corr.size == 0:
                                    missing_items.append(f"{row['analysis_id']} | {roi} | {direction} -> no finite points")
                                    continue

                                x_res = _transform_x(
                                    xvar=xvar,
                                    active_corr=x_corr,
                                    fit_row=row,
                                    peak_D0_fix=peak_D0_fix,
                                    peak_gamma=peak_gamma,
                                )
                                m_res = np.isfinite(x_res) & np.isfinite(y)
                                if not np.any(m_res):
                                    missing_items.append(f"{row['analysis_id']} | {roi} | {direction} -> xvar={xvar} has no valid points")
                                    continue
                                curve_x_plot = x_res[m_res]
                                curve_y_plot = y[m_res]
                                order = np.argsort(curve_x_plot)
                                x_bounds.append(curve_x_plot)
                                ax.plot(
                                    curve_x_plot[order],
                                    curve_y_plot[order],
                                    "-",
                                    color=color,
                                    linewidth=2.1,
                                    alpha=0.95,
                                )

                            else:
                                if show_data_points:
                                    contrast_path = _resolve_contrast_parquet_for_row(
                                        row,
                                        contrast_root=contrast_root,
                                        cache=cache,
                                    )
                                    print(
                                        "[diagnostic] loading contrast table "
                                        f"analysis_id={row['analysis_id']} sheet={row.get('sheet', None)} "
                                        f"roi={roi} direction={direction} td_ms={row.get('td_ms', None)} "
                                        f"N=({row.get('N_1', None)}, {row.get('N_2', None)}) path={contrast_path}"
                                    )
                                    contrast_df = _load_contrast_table_cached(contrast_path, cache)
                                    before_group = len(contrast_df)
                                    df_group = _subset_group(contrast_df, row)
                                    after_group = len(df_group)
                                    print(
                                        "[diagnostic] contrast points "
                                        f"before_filter={before_group} after_filter={after_group}"
                                    )
                                    if df_group.empty:
                                        missing_items.append(f"{row['analysis_id']} | {roi} | {direction} -> empty group")
                                        continue

                                    x_corr, y, G1, G2 = _extract_plot_arrays(df_group, row)
                                    if x_corr.size == 0:
                                        missing_items.append(f"{row['analysis_id']} | {roi} | {direction} -> no finite points")
                                        continue

                                    x_plot = _transform_x(
                                        xvar=xvar,
                                        active_corr=x_corr,
                                        fit_row=row,
                                        peak_D0_fix=peak_D0_fix,
                                        peak_gamma=peak_gamma,
                                    )
                                    m_plot = np.isfinite(x_plot) & np.isfinite(y)
                                    if not np.any(m_plot):
                                        missing_items.append(f"{row['analysis_id']} | {roi} | {direction} -> xvar={xvar} has no valid points")
                                        continue
                                    x_data = x_plot[m_plot]
                                    y_data = y[m_plot]
                                    x_bounds.append(x_data)
                                    ax.plot(x_data, y_data, "o", color=color, markersize=4, alpha=0.95)

                                    xs_corr, ys = _build_fit_curve(row, G1, G2)
                                    xs_plot = _transform_x(
                                        xvar=xvar,
                                        active_corr=xs_corr,
                                        fit_row=row,
                                        peak_D0_fix=peak_D0_fix,
                                        peak_gamma=peak_gamma,
                                    )
                                    m_fit = np.isfinite(xs_plot) & np.isfinite(ys)
                                    if np.any(m_fit):
                                        x_fit = xs_plot[m_fit]
                                        y_fit = ys[m_fit]
                                        order = np.argsort(x_fit)
                                        x_bounds.append(x_fit)
                                        ax.plot(
                                            x_fit[order],
                                            y_fit[order],
                                            "--" if show_resampled_fit else "-",
                                            color=color,
                                            linewidth=1.3 if show_resampled_fit else 1.8,
                                            alpha=0.35 if show_resampled_fit else 0.95,
                                        )

                                if show_resampled_fit:
                                    g_res, y_res = _build_resampled_fit_curve(row, n_points=int(resampled_curve_n))
                                    x_res = _transform_x(
                                        xvar=xvar,
                                        active_corr=g_res,
                                        fit_row=row,
                                        peak_D0_fix=peak_D0_fix,
                                        peak_gamma=peak_gamma,
                                    )
                                    m_res = np.isfinite(x_res) & np.isfinite(y_res)
                                    if np.any(m_res):
                                        curve_x_plot = x_res[m_res]
                                        curve_y_plot = y_res[m_res]
                                        order = np.argsort(curve_x_plot)
                                        x_bounds.append(curve_x_plot)
                                        ax.plot(
                                            curve_x_plot[order],
                                            curve_y_plot[order],
                                            "-",
                                            color=color,
                                            linewidth=2.1,
                                            alpha=0.95,
                                        )

                            if xvar in peak_marker_x_vars:
                                marker_sources = ("standard", "resampled") if peak_source == "both" else (peak_source,)
                                for marker_source in marker_sources:
                                    if contrast_source == "fitted_resampled" and marker_source == "resampled":
                                        x_peak, y_peak = _peak_point_from_curve(
                                            x_values=curve_x_plot,
                                            y_values=curve_y_plot,
                                        )
                                        tc_peak_ms = x_peak if xvar == "tc" else float(pd.to_numeric(pd.Series([row.get("tc_peak_resampled_ms", np.nan)]), errors="coerce").iloc[0])
                                    else:
                                        x_peak, y_peak = _peak_point_for_xvar(
                                            row,
                                            xvar=xvar,
                                            peak_D0_fix=peak_D0_fix,
                                            peak_gamma=peak_gamma,
                                            peak_source=marker_source,
                                        )
                                        tc_col = "tc_peak_resampled_ms" if marker_source == "resampled" else "tc_peak_ms"
                                        tc_peak_ms = float(pd.to_numeric(pd.Series([row.get(tc_col, np.nan)]), errors="coerce").iloc[0])
                                    if not (np.isfinite(x_peak) and np.isfinite(y_peak)):
                                        continue
                                    marker = "D" if marker_source == "resampled" else "o"
                                    ax.scatter(
                                        [x_peak],
                                        [y_peak],
                                        s=42 if marker_source == "resampled" else 34,
                                        marker=marker,
                                        color=color,
                                        edgecolors="black",
                                        linewidths=0.7,
                                        zorder=5 if marker_source == "resampled" else 4,
                                    )
                                    if np.isfinite(tc_peak_ms):
                                        label = "r " if marker_source == "resampled" else ""
                                        ax.annotate(
                                            f"{label}{tc_peak_ms:.1f} ms",
                                            (x_peak, y_peak),
                                            xytext=(4, 4),
                                            textcoords="offset points",
                                            fontsize=6.5,
                                            color=color,
                                            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.65),
                                        )
                        except Exception as exc:
                            missing_items.append(f"{row['analysis_id']} | {roi} | {direction} -> {exc}")

                    if x_bounds:
                        x_all = np.concatenate([arr[np.isfinite(arr)] for arr in x_bounds if arr.size > 0])
                        if x_all.size > 0:
                            xmin = float(np.nanmin(x_all))
                            xmax = float(np.nanmax(x_all))
                            if np.isfinite(xmin) and np.isfinite(xmax):
                                pad = 0.05 * (xmax - xmin) if xmax > xmin else max(1.0, 0.05 * abs(xmax) if xmax != 0.0 else 1.0)
                                ax.set_xlim(xmin - pad, xmax + pad)
                    if xvar in x_lims:
                        ax.set_xlim(*x_lims[xvar])

                    ax.set_title(f"{roi} | {direction}", fontsize=11)
                    ax.grid(True, alpha=0.25)
                    if i == len(roi_order) - 1:
                        ax.set_xlabel(_x_label(spec.gbase, spec.xplot, xvar, use_resampled_gradient=use_resampled_gradient), fontsize=11)
                    if j == 0:
                        ax.set_ylabel(spec.ycol, fontsize=11)

            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    color=td_to_color[float(td)],
                    marker="o",
                    linewidth=2,
                    markersize=5,
                    markeredgecolor="black",
                    markeredgewidth=0.4,
                    label=f"td={float(td):g} ms",
                )
                for td in td_values
            ]
            fig.legend(
                handles=legend_handles,
                loc="center right",
                bbox_to_anchor=(0.995, 0.5),
                title="Curves",
                fontsize=9,
                title_fontsize=10,
            )

            peak_note = f" | annotated peaks = {peak_source}" if xvar in peak_marker_x_vars else ""
            gradient_note = " | gradient=g_resampled" if use_resampled_gradient else ""
            fit_note = " | solid fit = resampled g" if show_resampled_fit and not use_resampled_gradient else ""
            fig.suptitle(
                f"{spec.subj} | model={spec.model} | y={spec.ycol} | x={xvar}{gradient_note}{peak_note}{fit_note}",
                fontsize=14,
            )
            plt.tight_layout(rect=[0.02, 0.02, 0.9, 0.95])

            gradient_token = _gradient_output_token(spec.xplot, use_resampled_gradient=use_resampled_gradient)
            out_path = out_dir / (
                f"contrast_tc_peak_panels_subj={_sanitize_token(spec.subj)}"
                f"_model={_sanitize_token(spec.model)}"
                f"_g={_sanitize_token(spec.gbase)}"
                f"_y={_sanitize_token(spec.ycol)}"
                f"_{gradient_token}"
                f"_xvar={_sanitize_token(xvar)}.png"
            )
            if use_resampled_gradient:
                stale_path = out_dir / (
                    f"contrast_tc_peak_panels_subj={_sanitize_token(spec.subj)}"
                    f"_model={_sanitize_token(spec.model)}"
                    f"_g={_sanitize_token(spec.gbase)}"
                    f"_y={_sanitize_token(spec.ycol)}"
                    f"_xplot={_sanitize_token(spec.xplot)}"
                    f"_xvar={_sanitize_token(xvar)}.png"
                )
                if stale_path != out_path and stale_path.exists():
                    stale_path.unlink()
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            outputs.append(out_path)

            if missing_items:
                warnings.warn(
                    "Some panels/curves could not be plotted:\n" + "\n".join(missing_items[:15]),
                    stacklevel=2,
                )

    if not outputs:
        raise ValueError("Could not generate any figure with the current filters.")

    return outputs
