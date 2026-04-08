from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from tools.brain_labels import infer_subj_label


# ---------------------------
# Modelo pseudo-huber (reparam)
# ---------------------------
def tc_pseudohuber(Td: np.ndarray, c: float, delta: float, alpha_macro: float) -> np.ndarray:
    Td = np.asarray(Td, float)
    delta = float(delta)
    return c + alpha_macro * delta * (np.sqrt(1.0 + (Td / delta) ** 2) - 1.0)

def alpha_of_Td(Td: np.ndarray, delta: float, alpha_macro: float) -> np.ndarray:
    Td = np.asarray(Td, float)
    return alpha_macro * Td / np.sqrt(delta**2 + Td**2)

def tc_quadratic_smallTd(Td: np.ndarray, c: float, delta: float, alpha_macro: float) -> np.ndarray:
    Td = np.asarray(Td, float)
    q = alpha_macro / (2.0 * delta)
    return c + q * Td**2

def tc_linear_largeTd(Td: np.ndarray, c: float, delta: float, alpha_macro: float) -> np.ndarray:
    Td = np.asarray(Td, float)
    return c - alpha_macro * delta + alpha_macro * Td

def A_from_params(delta: float, alpha_macro: float) -> float:
    return float(alpha_macro / delta) if delta > 0 else float("nan")

def qquad_from_params(delta: float, alpha_macro: float) -> float:
    return float(alpha_macro / (2.0 * delta)) if delta > 0 else float("nan")

def qquad_se(delta: float, alpha_macro: float, delta_se: float, alpha_se: float) -> float:
    if not np.isfinite(delta) or delta <= 0:
        return float("nan")
    if not np.isfinite(delta_se) or not np.isfinite(alpha_se):
        return float("nan")
    dqda = 1.0 / (2.0 * delta)
    dqdd = -alpha_macro / (2.0 * delta**2)
    var = (dqda**2) * (alpha_se**2) + (dqdd**2) * (delta_se**2)
    return float(np.sqrt(var))

def tc_pseudohuber_alpha_fixed(Td: np.ndarray, c: float, delta: float, alpha_fixed: float) -> np.ndarray:
    return tc_pseudohuber(Td, c, delta, alpha_fixed)


# ===========================
# Helpers genéricos
# ===========================
def _as_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def _ensure_required_cols(df: pd.DataFrame, cols: list[str], where: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"{where}: faltan columnas {miss}. Tengo: {list(df.columns)}")

def _ensure_subj(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura columna 'subj' siempre presente y no vacía.
    Si no existe, intenta derivar desde 'Archivo_origen' o 'source_file'.
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
    # ordenar dejando preferred al principio si existen
    pref = [d for d in preferred if d in dirs]
    rest = [d for d in dirs if d not in pref]
    return pref + rest

def _subset_last(x: np.ndarray, y: np.ndarray, k_last: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if k_last is None or len(x) <= k_last:
        return x, y
    return x[-k_last:], y[-k_last:]

def _fit_least_squares(fun, p0, bounds):
    res = least_squares(fun, x0=p0, bounds=bounds)
    p = res.x
    se = np.full_like(p, np.nan, dtype=float)
    if res.jac is not None:
        J = res.jac
        dof = max(0, J.shape[0] - J.shape[1])
        if dof > 0:
            s2 = float(res.cost * 2 / dof)  # cost=0.5*sum(r^2)
            try:
                cov = np.linalg.inv(J.T @ J) * s2
                se = np.sqrt(np.diag(cov))
            except np.linalg.LinAlgError:
                pass
    return p, se, res

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
# Summary alpha_macro loader (GENÉRICO)
# ---------------------------
def load_alpha_macro_summary(summary_xlsx: Path) -> pd.DataFrame:
    """
    Lee summary_alpha_values.xlsx de forma tolerante.

    Espera columnas parecidas a:
      subj, region, direccion, alpha, alpha_error (nombres libres)

    Devuelve SIEMPRE:
      subj, roi, direction, alpha_macro, alpha_macro_error

    - Mantiene TODAS las direcciones presentes (ej: '1','2','3', 'x','y','z', 'long','tra', ...).
    - Si detecta direcciones x/y/z, agrega derivadas:
        long := x
        tra  := mean(y,z)
      (solo si no existen ya en el archivo)
    """
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
        raise KeyError(f"load_alpha_macro_summary: faltan columnas {missing}. Tengo: {list(raw.columns)}")

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

    # Derivadas long/tra si hay x/y/z
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


# ---------------------------
# Block helpers
# ---------------------------
def _region2color(regiones: list[str], palette: list[str]) -> Dict[str, str]:
    return {r: palette[i % len(palette)] for i, r in enumerate(regiones)}

def _shade(color: str, factor: float):
    import matplotlib.colors as mcolors
    rgb = mcolors.to_rgb(color)
    white = (1, 1, 1)
    return tuple(white[i] + factor * (rgb[i] - white[i]) for i in range(3))

def _markers_for_subjs(subjs: list[str]) -> Dict[str, str]:
    mk = ["o","s","^","D","v","P","X","*","<",">"]
    return {s: mk[i % len(mk)] for i, s in enumerate(subjs)}


# ---------------------------
# BLOQUE 1: fit + plots tc(Td)
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
) -> pd.DataFrame:
    """
    mode:
      - free_macro: ajusta (c, delta, alpha_macro)
      - fixed_macro: fija alpha_macro = alpha_macro(summary) y ajusta (c, delta)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df_params.copy()
    df["roi"] = df["roi"].astype(str).str.replace("_norm", "", regex=False)
    df = _ensure_subj(df)
    df = _ensure_direction(df)

    _ensure_required_cols(df, ["td_ms", y_col], "fit_tc_vs_td_pseudohuber")

    regiones = [r.replace("_norm","") for r in cfg_regions if r.replace("_norm","") in df["roi"].unique()]
    if not regiones:
        regiones = sorted(df["roi"].unique())

    subjs = sorted(df["subj"].unique())
    region2color = _region2color(regiones, palette)
    markers = _markers_for_subjs(subjs)

    rows = []

    dirs = _directions_present(df)
    if not dirs:
        raise ValueError("No hay valores en la columna 'direction' de groupfits.")

    # Normalizar alpha_macro_df si se usa
    if alpha_macro_df is not None:
        alpha_macro_df = alpha_macro_df.copy()
        alpha_macro_df["roi"] = alpha_macro_df["roi"].astype(str).str.replace("_norm", "", regex=False)
        alpha_macro_df = _ensure_subj(alpha_macro_df)
        alpha_macro_df = _ensure_direction(alpha_macro_df)

    for dir_actual in dirs:
        df_dir = df[df["direction"] == dir_actual]

        for subj in subjs:
            for region in regiones:
                sub = df_dir[(df_dir["subj"] == subj) & (df_dir["roi"] == region)].sort_values("td_ms")
                if sub.empty:
                    continue

                x = sub["td_ms"].to_numpy(dtype=float)
                y = sub[y_col].to_numpy(dtype=float)

                # si hay NaNs, limpiarlos
                m = np.isfinite(x) & np.isfinite(y)
                x, y = x[m], y[m]
                if len(x) == 0:
                    continue

                x_fit, y_fit = _subset_last(x, y, k_last)

                if mode == "fixed_macro":
                    if alpha_macro_df is None:
                        continue

                    # dentro del loop, justo donde buscás alpha_fixed:
                    sheet_val = None
                    if "sheet" in sub.columns:
                        uu = sub["sheet"].dropna().astype(str).unique()
                        if len(uu) == 1:
                            sheet_val = uu[0]

                    # 1) intento por subj
                    mdf = alpha_macro_df[
                        (alpha_macro_df["subj"] == subj) &
                        (alpha_macro_df["roi"] == region) &
                        (alpha_macro_df["direction"] == dir_actual)
                    ]

                    # 2) fallback por sheet si no matchea por subj
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

                    # fit c, delta
                    c0 = float(np.min(y_fit))
                    delta0 = float(np.median(x_fit)) if np.median(x_fit) > 0 else 10.0
                    p0 = np.array([c0, delta0], float)
                    bounds = ([0.0, 1e-6], [np.inf, np.inf])

                    def fun(p):
                        c, delta = p
                        return tc_pseudohuber_alpha_fixed(x_fit, c, delta, alpha_fixed) - y_fit

                    p, se, res = _fit_least_squares(fun, p0, bounds)
                    c, delta = map(float, p)
                    c_se, delta_se = map(float, se)

                    alpha_macro = alpha_fixed
                    alpha_macro_se = alpha_fixed_err

                else:
                    # free alpha_macro
                    if len(x_fit) < 3:
                        # 3 parámetros -> al menos 3 puntos
                        continue

                    c0 = float(np.min(y_fit))
                    delta0 = float(np.median(x_fit)) if np.median(x_fit) > 0 else 10.0
                    alpha0 = 0.2
                    p0 = np.array([c0, delta0, alpha0], float)
                    alpha_min, alpha_max = 0.1, 0.3
                    bounds = ([-np.inf, 1e-6, alpha_min], [np.inf, np.inf, alpha_max])

                    def fun(p):
                        c, delta, alpha_macro = p
                        return tc_pseudohuber(x_fit, c, delta, alpha_macro) - y_fit

                    p, se, res = _fit_least_squares(fun, p0, bounds)
                    c, delta, alpha_macro = map(float, p)
                    c_se, delta_se, alpha_macro_se = map(float, se)

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
                    "A": A,
                    "q_quad": q_quad,
                    "q_quad_se": q_quad_se,
                    "r2": r2,
                })

        # Plot por regiones (grid) para esta direction
        if len(regiones) == 0:
            continue

        nrows, ncols = _make_grid(len(regiones), ncols=3)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.6*nrows), sharex=True, sharey=False)
        axes = np.array(axes).reshape(-1)

        for ax, region in zip(axes, regiones):
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

                # obtener params fit
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
            ax.grid(True)
            if any_line:
                ax.legend(fontsize=9)

        # limpiar axes vacíos si hay más ejes que regiones
        for ax in axes[len(regiones):]:
            ax.axis("off")

        plt.suptitle(f"PseudoHuber model fit | y={y_col} | dir={dir_actual} | mode={mode} | k_last={k_last}", fontsize=18)
        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(out_dir / f"BLOCK1_{y_col}_fit_dir={dir_actual}_mode={mode}_k={k_last}.png", dpi=300)
        plt.close()

    if not rows:
        raise ValueError(
            "No se generó ningún ajuste tc_vs_td (df_fit quedó vacío). "
            "Causas típicas: (i) no hay >=3 puntos Td por (subj, roi, direction) en modo free_macro; "
            "(ii) en fixed_macro falta alpha_macro para esas claves; "
            "(iii) Direcciones no coinciden entre groupfits y summary."
        )

    df_fit = pd.DataFrame(rows)
    df_fit.to_excel(out_dir / f"params_pseudohuber_mode={mode}_y={y_col}_k={k_last}.xlsx", index=False)
    return df_fit


# ---------------------------
# BLOQUE 2: plots vs regiones (alpha_macro y delta) + opcional A
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

    regiones = [r.replace("_norm","") for r in cfg_regions if r.replace("_norm","") in df_fit["roi"].unique()]
    if not regiones:
        regiones = sorted(df_fit["roi"].unique())

    subjs = sorted(df_fit["subj"].unique())
    dirs = _directions_present(df_fit)

    def plot_var(var: str, err: str, title: str, fname: str):
        for dir_actual in dirs:
            fig, ax = plt.subplots(1, 1, figsize=(12, 5))
            any_line = False

            for subj in subjs:
                sub = df_fit[(df_fit["direction"] == dir_actual) & (df_fit["subj"] == subj)]
                xs = np.arange(len(regiones))

                ys, es = [], []
                for r in regiones:
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

            ax.set_xticks(np.arange(len(regiones)))
            ax.set_xticklabels(regiones, rotation=45, ha="right")
            ax.set_title(f"{title} | dir={dir_actual}", fontsize=16)
            ax.grid(True)
            if any_line:
                ax.legend()
            plt.tight_layout()
            plt.savefig(out_dir / f"{fname}_dir={dir_actual}.png", dpi=300)
            plt.close()

    plot_var("alpha_macro", "alpha_macro_se", r"$\alpha_{macro} = A\delta$", "BLOCK2_alpha_macro_vs_region")
    plot_var("delta", "delta_se", r"$\delta$", "BLOCK2_delta_vs_region")
    plot_var("c", "c_se", r"$c$", "BLOCK2_c_vs_region")
    plot_var("sqrt_q", "sqrt_q_se", r"$\sqrt{q}$", "BLOCK2_sqrt_q_vs_region")
    plot_var("q_quad", "q_quad_se", r"$q=\alpha_{macro}/(2\delta)$", "BLOCK2_qquad_vs_region")
    if plot_A:
        plot_var("A", "A_se", r"$A$", "BLOCK2_A_vs_region")


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
    palette: list[str],  # mantenemos API
    *,
    show_errorbars: bool = True,
    tag: str | None = None,
    fname: str | None = None,
) -> None:
    """
    Versión GENÉRICA del plot comparativo por direcciones:
    - Antes: 2 columnas (long/tra)
    - Ahora: 1xN columnas para TODAS las direcciones presentes (ordenando long/tra primero si existen)
    - Misma escala Y por fila (sharey='row')
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df_fit = _ensure_alpha_macro_cols(df_fit.copy())
    df_fit = _ensure_subj(df_fit)
    df_fit = _ensure_direction(df_fit)
    df_fit = _ensure_A_se(df_fit)
    df_fit = _ensure_sqrt_q_cols(df_fit)

    regiones = [r.replace("_norm", "") for r in cfg_regions]
    regiones = [r for r in regiones if r in df_fit["roi"].unique()]
    if not regiones:
        regiones = sorted(df_fit["roi"].unique())

    subjs = sorted(df_fit["subj"].unique())
    markers = _markers_for_subjs(subjs)

    directions = _directions_present(df_fit)  # dinámico
    if not directions:
        print("[INFO] block2b: no hay direcciones -> skip.")
        return

    x = np.arange(len(regiones))

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

    # normalizar axes a 2D
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
                sub = df_dir[df_dir["subj"] == subj].set_index("roi").reindex(regiones)
                y = sub[var].to_numpy(dtype=float) if var in sub.columns else np.full(len(regiones), np.nan)
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

    # X ticks solo abajo
    for ax in axes[-1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(regiones, rotation=25, ha="right", fontsize=10)

    # Fijar límites Y por fila usando TODOS los puntos
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
        fname = f"BLOCK2b_vars_sameY_dirs={dirs_tag}_{tag}.png"

    fig.suptitle("Pseudo-Huber: q (Taylor), $\\alpha_{macro}$, $\\delta$, A, c, $\\sqrt{q}$ vs regiones", fontsize=14)
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(out_dir / fname, dpi=300)
    plt.close()


# ---------------------------
# BLOQUE 3: alpha_macro summary vs alpha_macro pseudo-Huber
# ---------------------------
def block3_alpha_macro_summary_vs_fit(
    df_fit: pd.DataFrame,
    out_dir: Path,
    alpha_macro_df: pd.DataFrame,
    palette: list[str],
    method_tag: str,
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
        print("[INFO] No hay intersección entre pseudo-huber fits y summary alpha_macro -> no se hace Block3.")
        return

    regiones = sorted(dfm["roi"].unique())
    region2color = _region2color(regiones, palette)

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

    plt.suptitle(f"BLOCK3 alpha_macro summary vs fit | {method_tag}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_dir / f"BLOCK3_alpha_macro_summary_vs_fit_{method_tag}.png", dpi=400)
    plt.close()


# ---------------------------
# BLOQUE 1b / 1c: ahora GENÉRICOS por direction
# ---------------------------
def block1b_alpha_vs_Td(df_params: pd.DataFrame, df_fit: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df_params = _ensure_subj(_ensure_direction(df_params.copy()))
    df_fit = _ensure_alpha_macro_cols(_ensure_subj(_ensure_direction(df_fit.copy())))

    subjs = sorted(df_fit["subj"].unique())
    regiones = sorted(df_fit["roi"].unique())
    directions = _directions_present(df_fit)

    for dir_actual in directions:
        # grid adaptable
        nrows, ncols = _make_grid(len(regiones), ncols=3)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.6*nrows), sharex=True)
        axes = np.array(axes).reshape(-1)

        for ax, region in zip(axes, regiones):
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

                xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                alpha_curve = alpha_of_Td(xx, delta, alpha_macro)
                alpha_small = A * xx
                alpha_asym = np.full_like(xx, alpha_macro)

                ax.plot(xx, alpha_curve, "-", linewidth=2, label=f"{subj} alpha(Td)")
                ax.plot(xx, alpha_small, "--", linewidth=1.5, label=f"{subj} A*Td")
                ax.plot(xx, alpha_asym, ":", linewidth=1.5, label=f"{subj} alpha_macro")
                any_line = True

            ax.set_title(region)
            ax.set_xlabel("Td [ms]")
            ax.set_ylabel("alpha(Td) = dtc/dTd")
            ax.grid(True)
            if any_line:
                ax.legend(fontsize=9)

        for ax in axes[len(regiones):]:
            ax.axis("off")

        plt.suptitle(f"BLOCK1b alpha(Td) + small/large-Td limits | dir={dir_actual}", fontsize=16)
        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(out_dir / f"BLOCK1b_alpha_vs_Td_dir={dir_actual}.png", dpi=300)
        plt.close()


def block1c_smallTd_tc_approx(
    df_params: pd.DataFrame,
    df_fit: pd.DataFrame,
    out_dir: Path,
    *,
    y_col: str = "tc_peak_ms",
    y_label: str = "$t_{c,peak}$ [ms]",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df_params = _ensure_subj(_ensure_direction(df_params.copy()))
    df_fit = _ensure_alpha_macro_cols(_ensure_subj(_ensure_direction(df_fit.copy())))

    subjs = sorted(df_fit["subj"].unique())
    regiones = sorted(df_fit["roi"].unique())
    directions = _directions_present(df_fit)

    for dir_actual in directions:
        nrows, ncols = _make_grid(len(regiones), ncols=3)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.6*nrows), sharex=True)
        axes = np.array(axes).reshape(-1)

        for ax, region in zip(axes, regiones):
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

                xx = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                y_full = tc_pseudohuber(xx, c, delta, alpha_macro)
                y_quad = tc_quadratic_smallTd(xx, c, delta, alpha_macro)

                ax.plot(x, y, "o", markersize=6, label=f"{subj} data")
                ax.plot(xx, y_full, "-", linewidth=2, label=f"{subj} full")
                ax.plot(xx, y_quad, "--", linewidth=1.5, label=f"{subj} quad small-Td")
                any_line = True

            ax.set_title(region)
            ax.set_xlabel("Td [ms]")
            ax.set_ylabel(y_label)
            ax.grid(True)
            if any_line:
                ax.legend(fontsize=9)

        for ax in axes[len(regiones):]:
            ax.axis("off")

        plt.suptitle(f"BLOCK1c {y_col}(Td) vs small-Td quadratic approx | dir={dir_actual}", fontsize=16)
        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.savefig(out_dir / f"BLOCK1c_{y_col}_smallTd_dir={dir_actual}.png", dpi=300)
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
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fit_dir = out_dir / "BLOCK1d_fullrange_per_fit"
    per_fit_dir.mkdir(parents=True, exist_ok=True)

    df_params = _ensure_subj(_ensure_direction(df_params.copy()))
    df_fit = _ensure_alpha_macro_cols(_ensure_subj(_ensure_direction(df_fit.copy())))

    subjs = sorted(df_fit["subj"].unique())
    regiones = sorted(df_fit["roi"].unique())
    directions = _directions_present(df_fit)
    xx = np.linspace(float(td_min_ms), float(td_max_ms), int(n_points))

    curve_rows = []

    for dir_actual in directions:
        nrows, ncols = _make_grid(len(regiones), ncols=3)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.8*nrows), sharex=True)
        axes = np.array(axes).reshape(-1)

        for ax, region in zip(axes, regiones):
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
                        f"BLOCK1d_{y_col}_subj={_safe_tag(subj)}"
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

        for ax in axes[len(regiones):]:
            ax.axis("off")

        plt.suptitle(
            f"BLOCK1d {y_col}(Td) full range + approximations | dir={dir_actual} | "
            f"{td_min_ms:.0f}-{td_max_ms:.0f} ms",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(out_dir / f"BLOCK1d_{y_col}_fullrange_dir={dir_actual}.png", dpi=300)
        plt.close()

    if curve_rows:
        df_curves = pd.DataFrame(curve_rows)
        df_curves.to_csv(out_dir / f"BLOCK1d_{y_col}_fullrange_curves.csv", index=False)
        df_curves.to_excel(out_dir / f"BLOCK1d_{y_col}_fullrange_curves.xlsx", index=False)


# ---------------------------
# BLOQUE 4: q vs alpha_macro (GENÉRICO)
# ---------------------------
def block4_qquad_vs_alpha_macro(
    df_fit: pd.DataFrame,
    out_dir: Path,
    alpha_macro_df: pd.DataFrame,
    palette: list[str],
    method_tag: str,
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
        print("[INFO] No hay intersección pseudo-huber vs summary -> no se hace Block4.")
        return

    regiones = sorted(dfm["roi"].unique())
    region2color = _region2color(regiones, palette)

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
            y = float(row["q_quad"])

            ax.plot(x, y, linestyle="None", marker=markers[v], markersize=9,
                    markerfacecolor=rgba, markeredgecolor=rgba)

            ax.text(
                x, y, region,
                fontsize=8, color="black", ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor=rgba, edgecolor="none", alpha=0.25),
            )

        ax.set_xlabel(r"$\alpha_{macro}$ summary", fontsize=16)
        if i == 0:
            ax.set_ylabel(r"$q=\alpha_{macro}/(2\delta)$", fontsize=16)
        ax.set_title(f"dir={dir_actual}", fontsize=14)
        ax.grid(True)

        handles = [
            Line2D([0], [0], marker=markers[v], linestyle="None", color="black", label=v, markersize=9)
            for v in volunteers
        ]
        ax.legend(handles=handles, title="Volunteer", fontsize=9, title_fontsize=10, loc="best")

    plt.suptitle(f"BLOCK4 q vs alpha_macro | {method_tag}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_dir / f"BLOCK4_q_vs_alpha_macro_{method_tag}.png", dpi=400)
    plt.close()
