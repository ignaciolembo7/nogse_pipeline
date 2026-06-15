from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Sequence, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_processing.io import write_xlsx_csv_outputs
from data_processing.master_table import select_alpha_macro
from fitting.core import least_squares_with_standard_errors
from tools.brain_labels import infer_subj_label
from tc_fittings.tc_td_models import (
    tc_pseudohuber,
    tc_linear,
    alpha_of_Td,
    tc_quadratic_smallTd,
    tc_linear_largeTd,
    A_from_params,
    qquad_from_params,
    qquad_se,
)

# ===========================
# Generic helpers
# ===========================
def _as_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def _ensure_required_cols(df: pd.DataFrame, cols: list[str], where: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"{where}: missing columns {miss}. Available columns: {list(df.columns)}")

def _ensure_subj(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the 'subj' column is always present and non-empty.
    When it is missing, try to derive it from 'Archivo_origen' or 'source_file'.
    """
    df = df.copy()
    if "subj" not in df.columns:
        if "Archivo_origen" in df.columns:
            df["subj"] = _as_str_series(df["Archivo_origen"])
        elif "source_file" in df.columns:
            df["subj"] = _as_str_series(df["source_file"]).apply(lambda s: infer_subj_label(None, source_name=s))
        else:
            df["subj"] = "UNKNOWN"
    df["subj"] = _as_str_series(df["subj"]).replace({"nan": "UNKNOWN", "None": "UNKNOWN", "": "UNKNOWN"})
    return df

def _ensure_direction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    _ensure_required_cols(df, ["direction"], "groupfits/df_fit")
    df["direction"] = _as_str_series(df["direction"])
    return df

def _ensure_alpha_macro_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "alpha_macro" not in df.columns and "alpha_inf" in df.columns:
        df = df.rename(columns={"alpha_inf": "alpha_macro"})
    if "alpha_macro_se" not in df.columns and "alpha_inf_se" in df.columns:
        df = df.rename(columns={"alpha_inf_se": "alpha_macro_se"})
    return df

def _directions_present(df: pd.DataFrame, preferred: tuple[str, ...] = ("long", "tra")) -> list[str]:
    df = _ensure_direction(df)
    dirs = [d for d in sorted(df["direction"].dropna().unique()) if d != ""]
    # Keep preferred directions first when they exist.
    pref = [d for d in preferred if d in dirs]
    rest = [d for d in dirs if d not in pref]
    return pref + rest

def _subset_last(x: np.ndarray, y: np.ndarray, k_last: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if k_last is None or len(x) <= k_last:
        return x, y
    return x[-k_last:], y[-k_last:]

def _fit_least_squares(fun, p0, bounds):
    return least_squares_with_standard_errors(fun, p0, bounds)


def _validate_fit_bound(name: str, lower: float, upper: float) -> None:
    if not np.isfinite(lower):
        raise ValueError(f"{name}_min must be finite.")
    if upper <= lower:
        raise ValueError(f"{name}_max must be greater than {name}_min.")


def _clip_initial(value: float, lower: float, upper: float) -> float:
    if np.isfinite(upper):
        return float(np.clip(value, lower, upper))
    return float(max(value, lower))

def _make_grid(n: int, ncols: int = 3) -> tuple[int, int]:
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols

def _safe_tag(text: object) -> str:
    return (
        str(text)
        .strip()
        .replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
    )


# ---------------------------
# Generic alpha_macro summary loader
# ---------------------------
def load_alpha_macro_summary(summary_xlsx: Path) -> pd.DataFrame:
    """
    Read summary_alpha_values.xlsx with tolerant column matching.

    Expected approximate columns:
      subj, region, direction, alpha, alpha_error (free-form names)

    Always returns:
      subj, roi, direction, alpha_macro, alpha_macro_error

    - Keeps all directions present, e.g. '1','2','3', 'x','y','z', 'long','tra', ...
    - When x/y/z directions are detected, add derived directions:
        long := x
        tra  := mean(y,z)
      only when they do not already exist in the file.
    """
    summary_xlsx = Path(summary_xlsx)
    suffix = summary_xlsx.suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(summary_xlsx)
    elif suffix == ".parquet":
        raw = pd.read_parquet(summary_xlsx)
    else:
        raw = pd.read_excel(summary_xlsx, decimal=",")
    lower_to_cols: dict[str, list[str]] = {}
    for col in raw.columns:
        lower_to_cols.setdefault(str(col).strip().lower(), []).append(col)

    def _pick(*aliases: str) -> pd.Series | None:
        for alias in aliases:
            cols = lower_to_cols.get(alias.lower(), [])
            if cols:
                return raw[cols[0]]
        return None

    subj = _pick("subj", "brain")
    roi = _pick("roi", "region")
    direction = _pick("direction", "direccion")
    alpha_macro = _pick("alpha_macro", "alpha")
    alpha_macro_error = _pick("alpha_macro_error", "alpha_error")
    sheet = _pick("sheet")

    if subj is None and sheet is not None:
        subj = _as_str_series(sheet).apply(lambda s: infer_subj_label(str(s), source_name=str(s)))

    missing = []
    if subj is None:
        missing.append("subj")
    if roi is None:
        missing.append("roi/region")
    if direction is None:
        missing.append("direction/direccion")
    if alpha_macro is None:
        missing.append("alpha_macro")
    if missing:
        raise KeyError(f"load_alpha_macro_summary: missing columns {missing}. Available columns: {list(raw.columns)}")

    df = pd.DataFrame(
        {
            "subj": subj,
            "roi": roi,
            "direction": direction,
            "alpha_macro": alpha_macro,
            "alpha_macro_error": alpha_macro_error if alpha_macro_error is not None else np.nan,
        }
    )
    if sheet is not None:
        df["sheet"] = sheet

    df["subj"] = _as_str_series(df["subj"])
    df["roi"] = _as_str_series(df["roi"]).str.replace("_norm", "", regex=False)
    df["direction"] = _as_str_series(df["direction"])
    df["alpha_macro"] = pd.to_numeric(df["alpha_macro"], errors="coerce")
    df["alpha_macro_error"] = pd.to_numeric(df["alpha_macro_error"], errors="coerce")
    if "sheet" in df.columns:
        df["sheet"] = _as_str_series(df["sheet"])

    base_cols = ["subj", "roi", "direction", "alpha_macro", "alpha_macro_error"]
    if "sheet" in df.columns:
        base_cols.insert(1, "sheet")
    base = df.dropna(subset=["alpha_macro"]).copy()
    base = base[base_cols]

    # Derive long/tra when x/y/z are available.
    dirs = set(base["direction"].unique())
    derived = []

    if ("x" in dirs) and ("long" not in dirs):
        dx = base[base["direction"] == "x"].copy()
        dx["direction"] = "long"
        derived.append(dx)

    if (("y" in dirs) or ("z" in dirs)) and ("tra" not in dirs):
        dyz = base[base["direction"].isin(["y", "z"])].copy()
        if not dyz.empty:
            group_cols = ["subj", "roi"]
            if "sheet" in dyz.columns:
                group_cols.insert(1, "sheet")
            dtra = dyz.groupby(group_cols, as_index=False).agg(
                alpha_macro=("alpha_macro", "mean"),
                alpha_macro_error=("alpha_macro_error", "mean"),
            )
            dtra["direction"] = "tra"
            keep = ["subj", "roi", "direction", "alpha_macro", "alpha_macro_error"]
            if "sheet" in dtra.columns:
                keep.insert(1, "sheet")
            derived.append(dtra[keep])

    out = pd.concat([base] + derived, ignore_index=True) if derived else base
    return out


def load_alpha_macro_table(
    table: pd.DataFrame,
    *,
    subjs: Sequence[str] | None = None,
    rois: Sequence[str] | None = None,
    directions: Sequence[str] | None = None,
    td_ms: float | None = None,
) -> pd.DataFrame:
    selectors: dict[str, object] = {}
    if subjs is not None:
        selectors["subj"] = [str(x) for x in subjs]
    if rois is not None:
        selectors["roi"] = [str(x).replace("_norm", "") for x in rois]
    if directions is not None:
        selectors["direction"] = [str(x) for x in directions]
    if td_ms is not None:
        selectors["td_ms"] = float(td_ms)
    return select_alpha_macro(table, **selectors)


# ---------------------------
# Block helpers
# ---------------------------
def _region2color(regions: list[str], palette: list[str]) -> Dict[str, str]:
    return {r: palette[i % len(palette)] for i, r in enumerate(regions)}

def _shade(color: str, factor: float):
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    white = (1, 1, 1)
    return tuple(white[i] + factor * (rgb[i] - white[i]) for i in range(3))

def _markers_for_subjs(subjs: list[str]) -> Dict[str, str]:
    mk = ["o","s","^","D","v","P","X","*","<",">"]
    return {s: mk[i % len(mk)] for i, s in enumerate(subjs)}


def _ordered_regions(region_order: list[str] | None, available: List[str]) -> list[str]:
    available_set = {str(r).replace("_norm", "") for r in available}
    if region_order:
        ordered: list[str] = []
        for region in region_order:
            r = str(region).replace("_norm", "")
            if r in available_set:
                ordered.append(r)
        if ordered:
            return ordered
    return sorted(available_set)


# ---------------------------
# Section 1: fit + plots tc(Td)
# ---------------------------
def fit_tc_vs_td_pseudohuber(
    *,
    df_params: pd.DataFrame,
    out_dir: Path,
    cfg_regions: list[str],
    palette: list[str],
    k_last: Optional[int],
    mode: str,  # "free_macro" | "fixed_macro"
    alpha_macro_df: Optional[pd.DataFrame] = None,
    y_col: str = "tc_peak_ms",
    y_label: str = "$t_{c,peak}$ [ms]",
    td_min_ms: float | None = None,
    td_max_ms: float | None = None,
    c_fixed: float | None = None,
    c_min: float = 0.0,
    c_max: float = np.inf,
    delta_fixed: float | None = None,
    delta_min: float = 1e-6,
    delta_max: float = np.inf,
    alpha_macro_fixed: float | None = None,
    alpha_macro_min: float = 0.1,
    alpha_macro_max: float = 0.3,
) -> pd.DataFrame:
    """
    mode:
      - free_macro: fit (c, delta, alpha_macro)
      - fixed_macro: fix alpha_macro = alpha_macro(summary) and fit (c, delta)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _validate_fit_bound("c", float(c_min), float(c_max))
    _validate_fit_bound("delta", float(delta_min), float(delta_max))
    _validate_fit_bound("alpha_macro", float(alpha_macro_min), float(alpha_macro_max))

    if c_fixed is not None and not np.isfinite(float(c_fixed)):
        raise ValueError("c_fixed must be finite.")
    if delta_fixed is not None and (not np.isfinite(float(delta_fixed)) or float(delta_fixed) <= 0):
        raise ValueError("delta_fixed must be finite and > 0.")
    if alpha_macro_fixed is not None and not np.isfinite(float(alpha_macro_fixed)):
        raise ValueError("alpha_macro_fixed must be finite.")

    df = df_params.copy()
    df["roi"] = df["roi"].astype(str).str.replace("_norm", "", regex=False)
    df = _ensure_subj(df)
    df = _ensure_direction(df)

    _ensure_required_cols(df, ["td_ms", y_col], "fit_tc_vs_td_pseudohuber")

    regions = [r.replace("_norm","") for r in cfg_regions if r.replace("_norm","") in df["roi"].unique()]
    if not regions:
        regions = sorted(df["roi"].unique())

    subjs = sorted(df["subj"].unique())
    region2color = _region2color(regions, palette)
    markers = _markers_for_subjs(subjs)

    rows = []

    dirs = _directions_present(df)
    if not dirs:
        raise ValueError("There are no values in the groupfits 'direction' column.")

    # Normalize alpha_macro_df when used.
    if alpha_macro_df is not None:
        alpha_macro_df = alpha_macro_df.copy()
        alpha_macro_df["roi"] = alpha_macro_df["roi"].astype(str).str.replace("_norm", "", regex=False)
        alpha_macro_df = _ensure_subj(alpha_macro_df)
        alpha_macro_df = _ensure_direction(alpha_macro_df)

    for dir_actual in dirs:
        df_dir = df[df["direction"] == dir_actual]

        for subj in subjs:
            for region in regions:
                sub = df_dir[(df_dir["subj"] == subj) & (df_dir["roi"] == region)].sort_values("td_ms")
                if sub.empty:
                    continue

                x = sub["td_ms"].to_numpy(dtype=float)
                y = sub[y_col].to_numpy(dtype=float)

                # Drop NaNs before fitting.
                m = np.isfinite(x) & np.isfinite(y)
                x, y = x[m], y[m]
                if len(x) == 0:
                    continue

                x_fit, y_fit = _subset_last(x, y, k_last)

                if mode == "fixed_macro":
                    if alpha_macro_df is None:
                        continue

                    # Resolve alpha_fixed for this loop entry.
                    sheet_val = None
                    if "sheet" in sub.columns:
                        uu = sub["sheet"].dropna().astype(str).unique()
                        if len(uu) == 1:
                            sheet_val = uu[0]

                    # First try matching by subject.
                    mdf = alpha_macro_df[
                        (alpha_macro_df["subj"] == subj) &
                        (alpha_macro_df["roi"] == region) &
                        (alpha_macro_df["direction"] == dir_actual)
                    ]

                    # Fall back to sheet when subject matching fails.
                    if mdf.empty and (sheet_val is not None) and ("sheet" in alpha_macro_df.columns):
                        mdf = alpha_macro_df[
                            (alpha_macro_df["sheet"].astype(str) == str(sheet_val)) &
                            (alpha_macro_df["roi"] == region) &
                            (alpha_macro_df["direction"] == dir_actual)
                        ]

                    if mdf.empty:
                        continue

                    alpha_fixed = float(mdf["alpha_macro"].values[0])
                    alpha_fixed_err = float(mdf.get("alpha_macro_error", pd.Series([float("nan")])).values[0])

                    alpha_fixed = float(mdf["alpha_macro"].values[0])
                    alpha_fixed_err = float(mdf.get("alpha_macro_error", pd.Series([np.nan])).values[0])

                    c0 = _clip_initial(float(np.min(y_fit)), float(c_min), float(c_max))
                    delta0 = _clip_initial(
                        float(np.median(x_fit)) if np.median(x_fit) > 0 else 10.0,
                        float(delta_min),
                        float(delta_max),
                    )
                    fit_names: list[str] = []
                    p0_parts: list[float] = []
                    lower_parts: list[float] = []
                    upper_parts: list[float] = []

                    if c_fixed is None:
                        fit_names.append("c")
                        p0_parts.append(c0)
                        lower_parts.append(float(c_min))
                        upper_parts.append(float(c_max))
                    if delta_fixed is None:
                        fit_names.append("delta")
                        p0_parts.append(delta0)
                        lower_parts.append(float(delta_min))
                        upper_parts.append(float(delta_max))

                    def unpack_fixed_macro(p):
                        vals = dict(zip(fit_names, p))
                        return (
                            float(c_fixed) if c_fixed is not None else float(vals["c"]),
                            float(delta_fixed) if delta_fixed is not None else float(vals["delta"]),
                        )

                    if fit_names:
                        if len(x_fit) < len(fit_names):
                            continue

                        def fun(p):
                            c_val, delta_val = unpack_fixed_macro(p)
                            return tc_pseudohuber(x_fit, c_val, delta_val, alpha_fixed) - y_fit

                        p, se, res = _fit_least_squares(
                            fun,
                            np.array(p0_parts, float),
                            (np.array(lower_parts, float), np.array(upper_parts, float)),
                        )
                        c, delta = unpack_fixed_macro(p)
                        se_by_name = dict(zip(fit_names, se))
                        c_se = float(se_by_name.get("c", 0.0 if c_fixed is not None else np.nan))
                        delta_se = float(se_by_name.get("delta", 0.0 if delta_fixed is not None else np.nan))
                    else:
                        c = float(c_fixed)
                        delta = float(delta_fixed)
                        c_se = 0.0
                        delta_se = 0.0

                    alpha_macro = alpha_fixed
                    alpha_macro_se = alpha_fixed_err

                else:
                    c0 = _clip_initial(float(np.min(y_fit)), float(c_min), float(c_max))
                    delta0 = _clip_initial(
                        float(np.median(x_fit)) if np.median(x_fit) > 0 else 10.0,
                        float(delta_min),
                        float(delta_max),
                    )
                    alpha0 = _clip_initial(0.2, float(alpha_macro_min), float(alpha_macro_max))
                    fit_names = []
                    p0_parts = []
                    lower_parts = []
                    upper_parts = []

                    if c_fixed is None:
                        fit_names.append("c")
                        p0_parts.append(c0)
                        lower_parts.append(float(c_min))
                        upper_parts.append(float(c_max))
                    if delta_fixed is None:
                        fit_names.append("delta")
                        p0_parts.append(delta0)
                        lower_parts.append(float(delta_min))
                        upper_parts.append(float(delta_max))
                    if alpha_macro_fixed is None:
                        fit_names.append("alpha_macro")
                        p0_parts.append(alpha0)
                        lower_parts.append(float(alpha_macro_min))
                        upper_parts.append(float(alpha_macro_max))

                    if len(x_fit) < max(1, len(fit_names)):
                        continue

                    def unpack_free_macro(p):
                        vals = dict(zip(fit_names, p))
                        return (
                            float(c_fixed) if c_fixed is not None else float(vals["c"]),
                            float(delta_fixed) if delta_fixed is not None else float(vals["delta"]),
                            float(alpha_macro_fixed) if alpha_macro_fixed is not None else float(vals["alpha_macro"]),
                        )

                    if fit_names:
                        def fun(p):
                            c_val, delta_val, alpha_val = unpack_free_macro(p)
                            return tc_pseudohuber(x_fit, c_val, delta_val, alpha_val) - y_fit

                        p, se, res = _fit_least_squares(
                            fun,
                            np.array(p0_parts, float),
                            (np.array(lower_parts, float), np.array(upper_parts, float)),
                        )
                        c, delta, alpha_macro = unpack_free_macro(p)
                        se_by_name = dict(zip(fit_names, se))
                        c_se = float(se_by_name.get("c", 0.0 if c_fixed is not None else np.nan))
                        delta_se = float(se_by_name.get("delta", 0.0 if delta_fixed is not None else np.nan))
                        alpha_macro_se = float(se_by_name.get("alpha_macro", 0.0 if alpha_macro_fixed is not None else np.nan))
                    else:
                        c = float(c_fixed)
                        delta = float(delta_fixed)
                        alpha_macro = float(alpha_macro_fixed)
                        c_se = 0.0
                        delta_se = 0.0
                        alpha_macro_se = 0.0

                # r2
                yhat = tc_pseudohuber(x_fit, c, delta, alpha_macro)
                ss_res = float(np.sum((y_fit - yhat) ** 2))
                ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

                A = A_from_params(delta, alpha_macro)
                q_quad = qquad_from_params(delta, alpha_macro)
                q_quad_se = qquad_se(delta, alpha_macro, delta_se, alpha_macro_se)

                rows.append({
                    "subj": subj,
                    "roi": region,
                    "direction": dir_actual,
                    "k_last": k_last,
                    "mode": mode,
                    "y_col": y_col,
                    "y_label": y_label,
                    "c": c, "c_se": c_se,
                    "delta": delta, "delta_se": delta_se,
                    "alpha_macro": alpha_macro, "alpha_macro_se": alpha_macro_se,
                    "c_fixed": c_fixed,
                    "c_min": c_min,
                    "c_max": c_max,
                    "delta_fixed": delta_fixed,
                    "delta_min": delta_min,
                    "delta_max": delta_max,
                    "alpha_macro_fixed": alpha_macro_fixed if mode != "fixed_macro" else np.nan,
                    "alpha_macro_min": alpha_macro_min,
                    "alpha_macro_max": alpha_macro_max,
                    "A": A,
                    "q_quad": q_quad,
                    "q_quad_se": q_quad_se,
                    "r2": r2,
                })

        # Plot regions as a grid for this direction.
        if len(regions) == 0:
            continue

        nrows, ncols = _make_grid(len(regions), ncols=3)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.6*nrows), sharex=True, sharey=False)
        axes = np.array(axes).reshape(-1)

        for ax, region in zip(axes, regions):
            base_color = region2color.get(region, "#1f77b4")
            any_line = False

            for i, subj in enumerate(subjs):
                sub = df_dir[(df_dir["subj"] == subj) & (df_dir["roi"] == region)].sort_values("td_ms")
                if sub.empty:
                    continue
                x = sub["td_ms"].to_numpy(float)
                y = sub[y_col].to_numpy(float)

                m = np.isfinite(x) & np.isfinite(y)
                x, y = x[m], y[m]
                if len(x) == 0:
                    continue

                # Retrieve fit parameters.
                fit_row = None
                for rr in rows[::-1]:
                    if rr["subj"] == subj and rr["roi"] == region and rr["direction"] == dir_actual:
                        fit_row = rr
                        break

                col = _shade(base_color, [0.25, 0.5, 1.0][i % 3])
                ax.plot(x, y, markers[subj], color=col, label=subj, markersize=7)
                any_line = True

                if fit_row is not None and len(x) >= 2:
                    xx = np.linspace(np.min(x), np.max(x), 200)
                    yy = tc_pseudohuber(xx, fit_row["c"], fit_row["delta"], fit_row["alpha_macro"])
                    ax.plot(xx, yy, "-", color=col, linewidth=2)

            ax.set_title(region, fontsize=14)
            ax.set_xlabel("Diffusion time $T_d$ [ms]", fontsize=16)
            ax.set_ylabel(y_label, fontsize=16)
            if td_min_ms is not None and td_max_ms is not None:
                ax.set_xlim(float(td_min_ms), float(td_max_ms))
            ax.grid(True)
            if any_line:
                ax.legend(fontsize=9)

        # Clear empty axes when there are more axes than regions.
        for ax in axes[len(regions):]:
            ax.axis("off")

        plt.suptitle(f"PseudoHuber model fit | y={y_col} | dir={dir_actual} | mode={mode} | k_last={k_last}", fontsize=18)
        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(out_dir / f"tc_td_{y_col}_fit_dir={dir_actual}_mode={mode}_k={k_last}.png", dpi=300)
        plt.close()

    if not rows:
        raise ValueError(
            "No tc_vs_td fit was generated (df_fit is empty). "
            "Typical causes: (i) there are not >=3 Td points per (subj, roi, direction) in free_macro mode; "
            "(ii) alpha_macro is missing for those keys in fixed_macro mode; "
            "(iii) directions do not match between groupfits and summary."
        )

    df_fit = pd.DataFrame(rows)
    write_xlsx_csv_outputs(df_fit, out_dir / f"params_pseudohuber_mode={mode}_y={y_col}_k={k_last}.xlsx")
    return df_fit


# ---------------------------
# Section 2: plots vs regions (alpha_macro and delta) + optional A
# ---------------------------
def block2_region_plots(
    df_fit: pd.DataFrame,
    out_dir: Path,
    cfg_regions: list[str],
    palette: list[str],
    plot_A: bool = True,
    show_errorbars: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df_fit = _ensure_alpha_macro_cols(df_fit.copy())
    df_fit = _ensure_subj(df_fit)
    df_fit = _ensure_direction(df_fit)
    df_fit = _ensure_sqrt_q_cols(df_fit)

    regions = [r.replace("_norm","") for r in cfg_regions if r.replace("_norm","") in df_fit["roi"].unique()]
    if not regions:
        regions = sorted(df_fit["roi"].unique())

    subjs = sorted(df_fit["subj"].unique())
    dirs = _directions_present(df_fit)

    def plot_var(var: str, err: str, title: str, fname: str):
        for dir_actual in dirs:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            any_line = False

            for subj in subjs:
                sub = df_fit[(df_fit["direction"] == dir_actual) & (df_fit["subj"] == subj)]
                xs = np.arange(len(regions))

                ys, es = [], []
                for r in regions:
                    row = sub[sub["roi"] == r]
                    if row.empty:
                        ys.append(np.nan); es.append(0.0)
                    else:
                        ys.append(float(row[var].values[0]) if var in row.columns else np.nan)
                        es.append(float(row[err].values[0]) if err in row.columns else 0.0)

                ys = np.array(ys, float); es = np.array(es, float)

                ax.plot(xs, ys, "o-", linewidth=2, markersize=7, label=subj)
                if show_errorbars:
                    ax.fill_between(xs, ys-es, ys+es, alpha=0.2)
                any_line = True

            ax.set_xticks(np.arange(len(regions)))
            ax.set_xticklabels(regions, rotation=45, ha="right")
            ax.set_title(f"{title} | dir={dir_actual}", fontsize=16)
            ax.grid(True)
            if any_line:
                ax.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"{fname}_dir={dir_actual}.png", dpi=300)
            plt.close()

    plot_var("alpha_macro", "alpha_macro_se", r"$\alpha_{macro} = A\delta$", "alpha_macro_vs_region")
    plot_var("delta", "delta_se", r"$\delta$", "delta_vs_region")
    plot_var("c", "c_se", r"$c$", "c_vs_region")
    plot_var("sqrt_q", "sqrt_q_se", r"$\sqrt{q}$", "sqrt_q_vs_region")
    plot_var("q_quad", "q_quad_se", r"$q=\alpha_{macro}/(2\delta)$", "qquad_vs_region")
    if plot_A:
        plot_var("A", "A_se", r"$A$", "A_vs_region")


def _ensure_A_se(df_fit: pd.DataFrame) -> pd.DataFrame:
    if "A_se" in df_fit.columns:
        return df_fit

    out = _ensure_alpha_macro_cols(df_fit.copy())
    A_se = np.full(len(out), np.nan, dtype=float)

    if ("alpha_macro_se" not in out.columns) or ("delta_se" not in out.columns):
        out["A_se"] = A_se
        return out

    for i, row in out.iterrows():
        delta = float(row.get("delta", np.nan))
        alpha = float(row.get("alpha_macro", np.nan))
        dse = float(row.get("delta_se", np.nan))
        ase = float(row.get("alpha_macro_se", np.nan))

        if not np.isfinite(delta) or delta <= 0 or not np.isfinite(alpha) or not np.isfinite(dse) or not np.isfinite(ase):
            continue

        dA_dalpha = 1.0 / delta
        dA_ddelta = -alpha / (delta**2)
        var = (dA_dalpha**2) * (ase**2) + (dA_ddelta**2) * (dse**2)
        A_se[i] = np.sqrt(var)

    out["A_se"] = A_se
    return out


def _ensure_sqrt_q_cols(df_fit: pd.DataFrame) -> pd.DataFrame:
    if "sqrt_q" in df_fit.columns and "sqrt_q_se" in df_fit.columns:
        return df_fit

    out = df_fit.copy()
    if "q_quad" not in out.columns:
        out["sqrt_q"] = np.nan
        out["sqrt_q_se"] = np.nan
        return out

    q = out["q_quad"].to_numpy(float)
    qse = out["q_quad_se"].to_numpy(float) if "q_quad_se" in out.columns else np.full_like(q, np.nan)
    sqrt_q = np.where(q > 0, np.sqrt(q), np.nan)
    sqrt_q_se = np.where((q > 0) & np.isfinite(qse) & (sqrt_q > 0), qse / (2.0 * sqrt_q), np.nan)

    out["sqrt_q"] = sqrt_q
    out["sqrt_q_se"] = sqrt_q_se
    return out


def block2b_cc_vars_long_tra_sameY(
    df_fit: pd.DataFrame,
    out_dir: Path,
    cfg_regions: list[str],
    palette: list[str],  # Keep API compatibility.
    *,
    show_errorbars: bool = True,
    tag: str | None = None,
    fname: str | None = None,
) -> None:
    """
    Generic comparative plot by direction:
    - Previously: 2 columns (long/tra)
    - Now: 1xN columns for every available direction, ordering long/tra first when present.
    - Same Y scale by row (sharey='row')
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df_fit = _ensure_alpha_macro_cols(df_fit.copy())
    df_fit = _ensure_subj(df_fit)
    df_fit = _ensure_direction(df_fit)
    df_fit = _ensure_A_se(df_fit)
    df_fit = _ensure_sqrt_q_cols(df_fit)

    regions = [r.replace("_norm", "") for r in cfg_regions]
    regions = [r for r in regions if r in df_fit["roi"].unique()]
    if not regions:
        regions = sorted(df_fit["roi"].unique())

    subjs = sorted(df_fit["subj"].unique())
    markers = _markers_for_subjs(subjs)

    directions = _directions_present(df_fit)  # Dynamic.
    if not directions:
        print("[INFO] var-grid plot: no directions found -> skip.")
        return

    x = np.arange(len(regions))

    specs = [
        ("q_quad", "q_quad_se", r"$q=\alpha_{macro}/(2\delta)$"),
        ("alpha_macro", "alpha_macro_se", r"$\alpha_{macro}$"),
        ("delta", "delta_se", r"$\delta$ [ms]"),
        ("A", "A_se", r"$A=\alpha_{macro}/\delta$"),
        ("c", "c_se", r"$c$"),
        ("sqrt_q", "sqrt_q_se", r"$\sqrt{q}$"),
    ]

    nrows = len(specs)
    ncols = len(directions)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3.0*nrows + 1), sharex=True, sharey="row")

    # Normalize axes to 2D.
    axes = np.array(axes)
    if nrows == 1 and ncols == 1:
        axes = axes.reshape((1, 1))
    elif nrows == 1:
        axes = axes.reshape((1, ncols))
    elif ncols == 1:
        axes = axes.reshape((nrows, 1))

    for col, dir_actual in enumerate(directions):
        df_dir = df_fit[df_fit["direction"] == dir_actual].copy()

        for row, (var, err, ylab) in enumerate(specs):
            ax = axes[row, col]
            any_line = False

            for subj in subjs:
                sub = df_dir[df_dir["subj"] == subj].set_index("roi").reindex(regions)
                y = sub[var].to_numpy(dtype=float) if var in sub.columns else np.full(len(regions), np.nan)
                if err in sub.columns:
                    e = sub[err].to_numpy(dtype=float)
                else:
                    e = np.zeros_like(y)

                if show_errorbars:
                    ax.errorbar(
                        x, y, yerr=e,
                        marker=markers.get(subj, "o"),
                        linestyle="-",
                        capsize=3,
                        label=subj,
                    )
                else:
                    ax.plot(
                        x, y,
                        marker=markers.get(subj, "o"),
                        linestyle="-",
                        label=subj,
                    )
                any_line = any_line or np.any(np.isfinite(y))

            ax.grid(True, alpha=0.3)
            ax.set_title(dir_actual, fontsize=12)
            if col == 0:
                ax.set_ylabel(ylab, fontsize=12)

            if row == 0 and col == 0 and any_line:
                ax.legend(fontsize=9, loc="best")

    # X ticks only on the bottom row.
    for ax in axes[-1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=25, ha="right", fontsize=10)

    # Set row-wise Y limits using all points.
    def _set_row_ylim(row_idx: int, var: str, err: str):
        yy = df_fit[var].to_numpy(dtype=float) if var in df_fit.columns else np.array([])
        ee = df_fit[err].to_numpy(dtype=float) if err in df_fit.columns else np.zeros_like(yy)
        m = np.isfinite(yy) & np.isfinite(ee)
        if yy.size == 0 or not np.any(m):
            return
        if show_errorbars:
            lo = np.min(yy[m] - ee[m])
            hi = np.max(yy[m] + ee[m])
        else:
            lo = np.min(yy[m])
            hi = np.max(yy[m])
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        for ax in axes[row_idx, :]:
            ax.set_ylim(lo - pad, hi + pad)

    for r, (var, err, _) in enumerate(specs):
        _set_row_ylim(r, var, err)

    if tag is None:
        mode = df_fit["mode"].unique() if "mode" in df_fit.columns else ["mixed"]
        kl   = df_fit["k_last"].unique() if "k_last" in df_fit.columns else ["?"]
        tag = f"mode={mode[0]}_k={kl[0]}" if (len(mode)==1 and len(kl)==1) else "mixed"

    if fname is None:
        dirs_tag = "_".join(directions)
        fname = f"vars_sameY_dirs={dirs_tag}_{tag}.png"

    fig.suptitle("Pseudo-Huber: q (Taylor), $\\alpha_{macro}$, $\\delta$, A, c, $\\sqrt{q}$ vs regions", fontsize=14)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_dir / fname, dpi=300)
    plt.close()


# ---------------------------
# Section 3: alpha_macro summary vs alpha_macro pseudo-Huber
# ---------------------------
def block3_alpha_macro_summary_vs_fit(
    df_fit: pd.DataFrame,
    out_dir: Path,
    alpha_macro_df: pd.DataFrame,
    palette: list[str],
    method_tag: str,
    region_order: list[str] | None = None,
) -> None:
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    out_dir.mkdir(parents=True, exist_ok=True)

    df_fit = _ensure_alpha_macro_cols(_ensure_subj(_ensure_direction(df_fit)))
    alpha_macro_df = _ensure_subj(_ensure_direction(alpha_macro_df))
    alpha_summary = alpha_macro_df.rename(
        columns={
            "alpha_macro": "alpha_macro_summary",
            "alpha_macro_error": "alpha_macro_summary_error",
        }
    )

    dfm = df_fit.merge(alpha_summary, on=["subj", "roi", "direction"], how="inner")
    if dfm.empty:
        print("[INFO] No overlap between pseudo-huber fits and summary alpha_macro -> skipping summary-vs-fit plot.")
        return

    regions = _ordered_regions(region_order, dfm["roi"].astype(str).unique().tolist())
    region2color = _region2color(regions, palette)

    volunteers = sorted(dfm["subj"].unique())
    markers = _markers_for_subjs(volunteers)

    directions = _directions_present(dfm)
    ncols = len(directions)
    fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 6), sharey=True)

    if ncols == 1:
        axes = [axes]

    for i, dir_actual in enumerate(directions):
        ax = axes[i]
        sub = dfm[dfm["direction"] == dir_actual]

        vpos = {v: j for j, v in enumerate(volunteers)}
        n = max(1, len(volunteers))

        for _, row in sub.iterrows():
            v = row["subj"]
            region = row["roi"]

            fade = vpos[v] / (n - 1) if n > 1 else 0.5
            base = region2color.get(region, "#000000")
            rgba = (*mcolors.to_rgb(base), 0.35 + 0.65 * (1 - fade))

            x = float(row["alpha_macro_summary"])
            y = float(row["alpha_macro"])

            ax.plot(x, y, linestyle="None", marker=markers[v], markersize=9,
                    markerfacecolor=rgba, markeredgecolor=rgba)

            ax.text(
                x, y, region,
                fontsize=8, color="black", ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=rgba, edgecolor="none", alpha=0.25),
            )

        ax.set_xlabel(r"$\alpha_{macro}$ summary", fontsize=16)
        if i == 0:
            ax.set_ylabel(r"$\alpha_{macro}$ pseudo-Huber", fontsize=16)
        ax.set_title(f"dir={dir_actual}", fontsize=14)
        ax.grid(True)

        handles = [
            Line2D([0], [0], marker=markers[v], linestyle="None", color="black", label=v, markersize=9)
            for v in volunteers
        ]
        ax.legend(handles=handles, title="Volunteer", fontsize=9, title_fontsize=10, loc="best")

    plt.suptitle(f"alpha_macro summary vs fit | {method_tag}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_dir / f"alpha_macro_summary_vs_fit_{method_tag}.png", dpi=400)
    plt.close()


# ---------------------------
# Section 1b / 1c: generic by direction
# ---------------------------
def block1b_alpha_vs_Td(
    df_params: pd.DataFrame,
    df_fit: pd.DataFrame,
    out_dir: Path,
    *,
    region_order: list[str] | None = None,
    td_min_ms: float | None = None,
    td_max_ms: float | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df_params = _ensure_subj(_ensure_direction(df_params.copy()))
    df_fit = _ensure_alpha_macro_cols(_ensure_subj(_ensure_direction(df_fit.copy())))

    subjs = sorted(df_fit["subj"].unique())
    regions = _ordered_regions(region_order, df_fit["roi"].astype(str).unique().tolist())
    directions = _directions_present(df_fit)

    for dir_actual in directions:
        # grid adaptable
        nrows, ncols = _make_grid(len(regions), ncols=3)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.6*nrows), sharex=True)
        axes = np.array(axes).reshape(-1)

        for ax, region in zip(axes, regions):
            any_line = False
            for subj in subjs:
                sub_data = df_params[
                    (df_params["direction"] == dir_actual) &
                    (df_params["subj"] == subj) &
                    (df_params["roi"] == region)
                ].sort_values("td_ms")

                sub_fit = df_fit[
                    (df_fit["direction"] == dir_actual) &
                    (df_fit["subj"] == subj) &
                    (df_fit["roi"] == region)
                ]
                if sub_data.empty or sub_fit.empty:
                    continue

                x = sub_data["td_ms"].to_numpy(float)
                c = float(sub_fit["c"].values[0])
                delta = float(sub_fit["delta"].values[0])
                alpha_macro = float(sub_fit["alpha_macro"].values[0])
                A = A_from_params(delta, alpha_macro)

                xmin = float(td_min_ms) if td_min_ms is not None else float(np.nanmin(x))
                xmax = float(td_max_ms) if td_max_ms is not None else float(np.nanmax(x))
                xx = np.linspace(xmin, xmax, 200)
                alpha_curve = alpha_of_Td(xx, delta, alpha_macro)
                alpha_small = A * xx

                ax.plot(xx, alpha_curve, "-", linewidth=2, label=f"{subj} alpha(Td)")
                ax.plot(xx, alpha_small, "--", linewidth=1.5, label=f"{subj} A*Td")
                any_line = True

            ax.set_title(region)
            ax.set_xlabel("Td [ms]")
            ax.set_ylabel("alpha(Td) = dtc/dTd")
            if td_min_ms is not None and td_max_ms is not None:
                ax.set_xlim(float(td_min_ms), float(td_max_ms))
            ax.grid(True)
            if any_line:
                ax.legend(fontsize=9)

        for ax in axes[len(regions):]:
            ax.axis("off")

        plt.suptitle(f"alpha(Td) + small-Td limit | dir={dir_actual}", fontsize=16)
        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(out_dir / f"alpha_vs_Td_dir={dir_actual}.png", dpi=300)
        plt.close()


def block1c_smallTd_tc_approx(
    df_params: pd.DataFrame,
    df_fit: pd.DataFrame,
    out_dir: Path,
    *,
    y_col: str = "tc_peak_ms",
    y_label: str = "$t_{c,peak}$ [ms]",
    region_order: list[str] | None = None,
    td_min_ms: float | None = None,
    td_max_ms: float | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df_params = _ensure_subj(_ensure_direction(df_params.copy()))
    df_fit = _ensure_alpha_macro_cols(_ensure_subj(_ensure_direction(df_fit.copy())))

    subjs = sorted(df_fit["subj"].unique())
    regions = _ordered_regions(region_order, df_fit["roi"].astype(str).unique().tolist())
    directions = _directions_present(df_fit)

    for dir_actual in directions:
        nrows, ncols = _make_grid(len(regions), ncols=3)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.6*nrows), sharex=True)
        axes = np.array(axes).reshape(-1)

        for ax, region in zip(axes, regions):
            any_line = False
            for subj in subjs:
                sub_data = df_params[
                    (df_params["direction"] == dir_actual) &
                    (df_params["subj"] == subj) &
                    (df_params["roi"] == region)
                ].sort_values("td_ms")

                sub_fit = df_fit[
                    (df_fit["direction"] == dir_actual) &
                    (df_fit["subj"] == subj) &
                    (df_fit["roi"] == region)
                ]
                if sub_data.empty or sub_fit.empty:
                    continue

                x = sub_data["td_ms"].to_numpy(float)
                y = sub_data[y_col].to_numpy(float)

                c = float(sub_fit["c"].values[0])
                delta = float(sub_fit["delta"].values[0])
                alpha_macro = float(sub_fit["alpha_macro"].values[0])

                xmin = float(td_min_ms) if td_min_ms is not None else float(np.nanmin(x))
                xmax = float(td_max_ms) if td_max_ms is not None else float(np.nanmax(x))
                xx = np.linspace(xmin, xmax, 200)
                y_full = tc_pseudohuber(xx, c, delta, alpha_macro)
                y_quad = tc_quadratic_smallTd(xx, c, delta, alpha_macro)

                ax.plot(x, y, "o", markersize=6, label=f"{subj} data")
                ax.plot(xx, y_full, "-", linewidth=2, label=f"{subj} full")
                ax.plot(xx, y_quad, "--", linewidth=1.5, label=f"{subj} quad small-Td")
                any_line = True

            ax.set_title(region)
            ax.set_xlabel("Td [ms]")
            ax.set_ylabel(y_label)
            if td_min_ms is not None and td_max_ms is not None:
                ax.set_xlim(float(td_min_ms), float(td_max_ms))
            ax.grid(True)
            if any_line:
                ax.legend(fontsize=9)

        for ax in axes[len(regions):]:
            ax.axis("off")

        plt.suptitle(f"{y_col}(Td) vs small-Td quadratic approximation | dir={dir_actual}", fontsize=16)
        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(out_dir / f"{y_col}_smallTd_dir={dir_actual}.png", dpi=300)
        plt.close()


def block1d_fullrange_tc_with_approximations(
    df_params: pd.DataFrame,
    df_fit: pd.DataFrame,
    out_dir: Path,
    *,
    y_col: str = "tc_peak_ms",
    y_label: str = "$t_{c,peak}$ [ms]",
    td_min_ms: float = 0.0,
    td_max_ms: float = 2000.0,
    n_points: int = 1000,
    region_order: list[str] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fit_dir = out_dir / "fullrange_per_fit"
    per_fit_dir.mkdir(parents=True, exist_ok=True)

    df_params = _ensure_subj(_ensure_direction(df_params.copy()))
    df_fit = _ensure_alpha_macro_cols(_ensure_subj(_ensure_direction(df_fit.copy())))

    subjs = sorted(df_fit["subj"].unique())
    regions = _ordered_regions(region_order, df_fit["roi"].astype(str).unique().tolist())
    directions = _directions_present(df_fit)
    xx = np.linspace(float(td_min_ms), float(td_max_ms), int(n_points))

    curve_rows = []

    for dir_actual in directions:
        nrows, ncols = _make_grid(len(regions), ncols=3)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.8*nrows), sharex=True)
        axes = np.array(axes).reshape(-1)

        for ax, region in zip(axes, regions):
            any_line = False
            for subj in subjs:
                sub_data = df_params[
                    (df_params["direction"] == dir_actual) &
                    (df_params["subj"] == subj) &
                    (df_params["roi"] == region)
                ].sort_values("td_ms")

                sub_fit = df_fit[
                    (df_fit["direction"] == dir_actual) &
                    (df_fit["subj"] == subj) &
                    (df_fit["roi"] == region)
                ]
                if sub_fit.empty:
                    continue

                c = float(sub_fit["c"].values[0])
                delta = float(sub_fit["delta"].values[0])
                alpha_macro = float(sub_fit["alpha_macro"].values[0])
                y_full = tc_pseudohuber(xx, c, delta, alpha_macro)
                y_quad = tc_quadratic_smallTd(xx, c, delta, alpha_macro)
                y_linear = tc_linear_largeTd(xx, c, delta, alpha_macro)

                curve_rows.extend(
                    {
                        "subj": subj,
                        "roi": region,
                        "direction": dir_actual,
                        "y_col": y_col,
                        "td_ms": float(xv),
                        "tc_full": float(yf),
                        "tc_quad_smallTd": float(yq),
                        "tc_linear_largeTd": float(yl),
                        "c": c,
                        "delta": delta,
                        "alpha_macro": alpha_macro,
                    }
                    for xv, yf, yq, yl in zip(xx, y_full, y_quad, y_linear)
                )

                label_base = f"{subj}"
                ax.plot(xx, y_full, "-", linewidth=2, label=f"{label_base} full")
                ax.plot(xx, y_quad, "--", linewidth=1.2, label=f"{label_base} quad")
                ax.plot(xx, y_linear, ":", linewidth=1.2, label=f"{label_base} linear")

                if not sub_data.empty:
                    x = sub_data["td_ms"].to_numpy(float)
                    y = sub_data[y_col].to_numpy(float)
                    m = np.isfinite(x) & np.isfinite(y)
                    if np.any(m):
                        ax.plot(x[m], y[m], "o", markersize=5, label=f"{label_base} data")

                any_line = True

                fig_one, ax_one = plt.subplots(1, 1, figsize=(8, 5))
                ax_one.plot(xx, y_full, "-", linewidth=2.5, label="full")
                ax_one.plot(xx, y_quad, "--", linewidth=1.8, label="quad small-Td")
                ax_one.plot(xx, y_linear, ":", linewidth=1.8, label="linear large-Td")
                if not sub_data.empty:
                    x = sub_data["td_ms"].to_numpy(float)
                    y = sub_data[y_col].to_numpy(float)
                    m = np.isfinite(x) & np.isfinite(y)
                    if np.any(m):
                        ax_one.plot(x[m], y[m], "o", markersize=6, label="data")
                ax_one.set_xlim(float(td_min_ms), float(td_max_ms))
                ax_one.set_xlabel("Td [ms]")
                ax_one.set_ylabel(y_label)
                ax_one.set_title(
                    f"{subj} | {region} | dir={dir_actual}\n"
                    f"c={c:.4g}, delta={delta:.4g} ms, alpha_macro={alpha_macro:.4g}"
                )
                ax_one.grid(True)
                ax_one.legend(fontsize=9)
                plt.tight_layout()
                plt.savefig(
                    per_fit_dir / (
                        f"{y_col}_subj={_safe_tag(subj)}"
                        f"_roi={_safe_tag(region)}_dir={_safe_tag(dir_actual)}.png"
                    ),
                    dpi=300,
                )
                plt.close(fig_one)

            ax.set_title(region)
            ax.set_xlim(float(td_min_ms), float(td_max_ms))
            ax.set_xlabel("Td [ms]")
            ax.set_ylabel(y_label)
            ax.grid(True)
            if any_line:
                ax.legend(fontsize=8)

        for ax in axes[len(regions):]:
            ax.axis("off")

        plt.suptitle(
            f"{y_col}(Td) full range + approximations | dir={dir_actual} | "
            f"{td_min_ms:.0f}-{td_max_ms:.0f} ms",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_dir / f"{y_col}_fullrange_dir={dir_actual}.png", dpi=300)
        plt.close()

    if curve_rows:
        df_curves = pd.DataFrame(curve_rows)
        write_xlsx_csv_outputs(
            df_curves,
            out_dir / f"{y_col}_fullrange_curves.xlsx",
            csv_path=out_dir / f"{y_col}_fullrange_curves.csv",
        )


# ---------------------------
# Section 4: q vs alpha_macro (generic)
# ---------------------------
def _block4_param_vs_alpha_macro(
    df_fit: pd.DataFrame,
    out_dir: Path,
    alpha_macro_df: pd.DataFrame,
    palette: list[str],
    method_tag: str,
    *,
    y_col: str,
    y_label: str,
    filename_prefix: str,
    skip_msg: str,
    region_order: list[str] | None = None,
) -> None:
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D

    out_dir.mkdir(parents=True, exist_ok=True)

    df_fit = _ensure_alpha_macro_cols(_ensure_subj(_ensure_direction(df_fit)))
    alpha_macro_df = _ensure_subj(_ensure_direction(alpha_macro_df))
    alpha_summary = alpha_macro_df.rename(
        columns={
            "alpha_macro": "alpha_macro_summary",
            "alpha_macro_error": "alpha_macro_summary_error",
        }
    )

    dfm = df_fit.merge(alpha_summary, on=["subj", "roi", "direction"], how="inner")
    if dfm.empty:
        print(f"[INFO] No overlap between pseudo-huber and summary -> skipping {skip_msg}.")
        return
    if y_col not in dfm.columns:
        print(f"[INFO] fit table has no {y_col!r} column -> skipping {skip_msg}.")
        return

    regions = _ordered_regions(region_order, dfm["roi"].astype(str).unique().tolist())
    region2color = _region2color(regions, palette)

    volunteers = sorted(dfm["subj"].unique())
    markers = _markers_for_subjs(volunteers)

    directions = _directions_present(dfm)
    ncols = len(directions)
    fig, axes = plt.subplots(1, ncols, figsize=(6*ncols, 6), sharey=True)
    if ncols == 1:
        axes = [axes]

    for i, dir_actual in enumerate(directions):
        ax = axes[i]
        sub = dfm[dfm["direction"] == dir_actual]
        vpos = {v: j for j, v in enumerate(volunteers)}
        n = max(1, len(volunteers))

        for _, row in sub.iterrows():
            v = row["subj"]
            region = row["roi"]
            fade = vpos[v] / (n - 1) if n > 1 else 0.5
            base = region2color.get(region, "#000000")
            rgba = (*mcolors.to_rgb(base), 0.35 + 0.65 * (1 - fade))
            x = float(row["alpha_macro_summary"])
            y = float(row[y_col])
            ax.plot(x, y, linestyle="None", marker=markers[v], markersize=9,
                    markerfacecolor=rgba, markeredgecolor=rgba)
            ax.text(
                x, y, region, fontsize=8, color="black", ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=rgba, edgecolor="none", alpha=0.25),
            )

        ax.set_xlabel(r"$\alpha_{macro}$ summary", fontsize=16)
        if i == 0:
            ax.set_ylabel(y_label, fontsize=16)
        ax.set_title(f"dir={dir_actual}", fontsize=14)
        ax.grid(True)
        handles = [
            Line2D([0], [0], marker=markers[v], linestyle="None", color="black", label=v, markersize=9)
            for v in volunteers
        ]
        ax.legend(handles=handles, title="Volunteer", fontsize=9, title_fontsize=10, loc="best")

    plt.suptitle(f"{filename_prefix} vs alpha_macro | {method_tag}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_dir / f"{filename_prefix}_vs_alpha_macro_{method_tag}.png", dpi=400)
    plt.close()


def block4_qquad_vs_alpha_macro(
    df_fit: pd.DataFrame,
    out_dir: Path,
    alpha_macro_df: pd.DataFrame,
    palette: list[str],
    method_tag: str,
    region_order: list[str] | None = None,
) -> None:
    _block4_param_vs_alpha_macro(
        df_fit, out_dir, alpha_macro_df, palette, method_tag,
        y_col="q_quad",
        y_label=r"$q=\alpha_{macro}/(2\delta)$",
        filename_prefix="q",
        skip_msg="q-vs-alpha plot",
        region_order=region_order,
    )


def block4_delta_vs_alpha_macro(
    df_fit: pd.DataFrame,
    out_dir: Path,
    alpha_macro_df: pd.DataFrame,
    palette: list[str],
    method_tag: str,
    region_order: list[str] | None = None,
) -> None:
    _block4_param_vs_alpha_macro(
        df_fit, out_dir, alpha_macro_df, palette, method_tag,
        y_col="delta",
        y_label=r"$\delta$ [ms]",
        filename_prefix="delta",
        skip_msg="delta-vs-alpha plot",
        region_order=region_order,
    )


# ---------------------------------------------------------------------------
# Generic tc vs Td fitting function
#
# Used by tc_td_registry to run any model registered in METHODS.
# For pseudohuber-specific features (fixed_macro mode, alpha vs Td plots,
# etc.) use fit_tc_vs_td_pseudohuber() directly.
# ---------------------------------------------------------------------------

def fit_tc_vs_td(
    df_params: pd.DataFrame,
    *,
    model_func: Callable,
    param_names: tuple[str, ...],
    param_inits: dict[str, float],
    param_bounds: dict[str, tuple[float, float]],
    fixed_params: dict[str, float] | None = None,
    k_last: int | None = None,
    y_col: str = "tc_peak_ms",
    y_label: str = r"$t_{c,peak}$ [ms]",
    td_min_ms: float | None = None,
    td_max_ms: float | None = None,
    out_dir: Path | None = None,
    cfg_regions: list[str] | None = None,
    palette: list[str] | None = None,
) -> pd.DataFrame:
    """
    Generic tc vs Td fitting loop for any model in tc_td_models.

    Parameters
    ----------
    model_func   : function(Td, *params) → np.ndarray
    param_names  : tuple of parameter names in the same order as model_func args
    param_inits  : initial guess for each parameter  {name: value}
    param_bounds : bounds for each parameter          {name: (lower, upper)}
    fixed_params : parameters held constant           {name: value}
                   (the rest are fitted freely)
    """
    fixed_params = fixed_params or {}
    free_names = [n for n in param_names if n not in fixed_params]

    df = df_params.copy()
    df["roi"] = df["roi"].astype(str).str.replace("_norm", "", regex=False)
    df = _ensure_subj(df)
    df = _ensure_direction(df)
    _ensure_required_cols(df, ["td_ms", y_col], "fit_tc_vs_td")

    if td_min_ms is not None:
        df = df[df["td_ms"] >= float(td_min_ms)]
    if td_max_ms is not None:
        df = df[df["td_ms"] <= float(td_max_ms)]

    regions_avail = sorted(df["roi"].unique())
    if cfg_regions:
        regions = [r.replace("_norm", "") for r in cfg_regions if r.replace("_norm", "") in regions_avail]
        if not regions:
            regions = regions_avail
    else:
        regions = regions_avail

    pal = palette or ["#a65628", "#e41a1c", "#ff7f00", "#984ea3", "#377eb8", "#999999"]
    region2color = _region2color(regions, pal)
    subjs = sorted(df["subj"].unique())
    markers = _markers_for_subjs(subjs)
    dirs = _directions_present(df)

    rows: list[dict] = []

    for dir_actual in dirs:
        df_dir = df[df["direction"] == dir_actual]
        nrows, ncols = _make_grid(len(regions), ncols=3)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.6 * nrows), sharex=True, sharey=False)
        axes = np.array(axes).reshape(-1)

        for ax, region in zip(axes, regions):
            base_color = region2color.get(region, "#1f77b4")
            any_data = False

            for i, subj in enumerate(subjs):
                sub = df_dir[(df_dir["subj"] == subj) & (df_dir["roi"] == region)].sort_values("td_ms")
                if sub.empty:
                    continue

                x = sub["td_ms"].to_numpy(dtype=float)
                y = sub[y_col].to_numpy(dtype=float)
                m = np.isfinite(x) & np.isfinite(y)
                x, y = x[m], y[m]
                if len(x) == 0:
                    continue

                x_fit, y_fit = _subset_last(x, y, k_last)
                if len(x_fit) < max(1, len(free_names)):
                    continue

                p0 = np.array([param_inits[n] for n in free_names], dtype=float)
                lo = np.array([param_bounds[n][0] for n in free_names], dtype=float)
                hi = np.array([param_bounds[n][1] for n in free_names], dtype=float)

                def _residuals(p, x_fit=x_fit, y_fit=y_fit):
                    param_vals = dict(fixed_params)
                    param_vals.update(zip(free_names, p))
                    args = [param_vals[n] for n in param_names]
                    return model_func(x_fit, *args) - y_fit

                p_fit, se_fit, _ = _fit_least_squares(_residuals, p0, (lo, hi))
                param_vals = dict(fixed_params)
                param_vals.update(zip(free_names, p_fit))
                se_vals = dict(zip(free_names, se_fit))

                args_fit = [param_vals[n] for n in param_names]
                yhat = model_func(x_fit, *args_fit)
                ss_res = float(np.sum((y_fit - yhat) ** 2))
                ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

                row: dict = {
                    "subj": subj,
                    "roi": region,
                    "direction": dir_actual,
                    "k_last": k_last,
                    "y_col": y_col,
                    "y_label": y_label,
                    "r2": r2,
                }
                for n in param_names:
                    row[n] = float(param_vals[n])
                    row[f"{n}_se"] = float(se_vals.get(n, 0.0 if n in fixed_params else float("nan")))
                rows.append(row)

                col = _shade(base_color, [0.25, 0.5, 1.0][i % 3])
                ax.plot(x, y, markers[subj], color=col, label=subj, markersize=7)
                any_data = True
                if len(x) >= 2:
                    xx = np.linspace(np.min(x), np.max(x), 200)
                    yy = model_func(xx, *args_fit)
                    ax.plot(xx, yy, "-", color=col, linewidth=2)

            ax.set_title(region, fontsize=14)
            ax.set_xlabel("Diffusion time $T_d$ [ms]", fontsize=16)
            ax.set_ylabel(y_label, fontsize=16)
            if td_min_ms is not None and td_max_ms is not None:
                ax.set_xlim(float(td_min_ms), float(td_max_ms))
            ax.grid(True)
            if any_data:
                ax.legend(fontsize=9)

        for ax in axes[len(regions):]:
            ax.axis("off")

        model_name = getattr(model_func, "__name__", "model")
        plt.suptitle(f"{model_name} fit | y={y_col} | dir={dir_actual} | k_last={k_last}", fontsize=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_dir / f"tc_td_{y_col}_fit_{model_name}_dir={dir_actual}_k={k_last}.png", dpi=300)
        plt.close()

    if not rows:
        raise ValueError(
            f"fit_tc_vs_td({model_func.__name__}): no rows produced. "
            "Check that df_params has enough (td_ms, y_col) points per (subj, roi, direction)."
        )

    df_fit = pd.DataFrame(rows)
    if out_dir is not None:
        model_name = getattr(model_func, "__name__", "model")
        write_xlsx_csv_outputs(df_fit, out_dir / f"params_{model_name}_y={y_col}_k={k_last}.xlsx")
    return df_fit
