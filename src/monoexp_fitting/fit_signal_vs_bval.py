from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from ogse_fitting.b_from_g import b_from_g
from plottings.fit_plot_style import finish_fit_figure, highlight_fit_points, plot_fit_curve, plot_fit_data, start_fit_figure
from tools.fit_params_schema import standardize_fit_params


def monoexp(b: np.ndarray, M0: float, D0: float) -> np.ndarray:
    return M0 * np.exp(-b * D0)


def monoexp_M0fixed(b: np.ndarray, D0: float, *, M0: float) -> np.ndarray:
    return M0 * np.exp(-b * D0)


def infer_exp_id(p: Path) -> str:
    name = p.name
    for suf in [".rot_tensor.long.parquet", ".long.parquet", ".parquet", ".xlsx", ".xls"]:
        if name.endswith(suf):
            return name[: -len(suf)]
    return p.stem


def _unique_float_any(df: pd.DataFrame, cols: Sequence[str], *, required: bool, name: str) -> Optional[float]:
    for c in cols:
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce").dropna().unique()
            if len(v) == 0:
                continue
            if len(v) != 1:
                raise ValueError(f"Esperaba 1 valor único en '{c}' para {name}, encontré: {v[:10]}")
            return float(v[0])
    if required:
        raise ValueError(f"No pude inferir {name}. Probé columnas: {list(cols)}")
    return None


def _unique_str(df: pd.DataFrame, col: str) -> Optional[str]:
    if col not in df.columns:
        return None
    u = pd.Series(df[col]).dropna().astype(str).unique()
    if len(u) == 1:
        return str(u[0])
    return None


def _normalize_requested_rois(
    *,
    roi: str = "ALL",
    rois: Optional[Sequence[str]] = None,
) -> Optional[list[str]]:
    if rois is not None:
        vals = [str(x) for x in rois]
        if any(v.upper() == "ALL" for v in vals):
            return None
        return vals

    roi_val = str(roi)
    if roi_val.upper() == "ALL":
        return None
    return [roi_val]


def _require_no_axis(df: pd.DataFrame) -> None:
    if "axis" in df.columns:
        raise ValueError("Encontré columna 'axis'. Este pipeline usa SOLO 'direction'.")


def _ensure_keys_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["direction"] = out["direction"].astype(str)
    out["roi"] = out["roi"].astype(str)
    bs = pd.to_numeric(out["b_step"], errors="coerce")
    if bs.isna().any():
        bad = out.loc[bs.isna(), ["roi", "direction", "b_step"]].head(10)
        raise ValueError(f"b_step inválido. Ejemplos:\n{bad.to_string(index=False)}")
    out["b_step"] = bs.astype(int)
    if "stat" in out.columns:
        out["stat"] = out["stat"].astype(str)
    return out


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _chi2(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sum((y - yhat) ** 2))


def _b_from_mode(
    d: pd.DataFrame,
    *,
    g_type: str,
    gamma: float,
    N: Optional[float],
    delta_ms: Optional[float],
    Delta_app_ms: Optional[float],
) -> np.ndarray:
    if g_type == "bvalue":
        if "bvalue" not in d.columns:
            raise ValueError("Falta columna 'bvalue'.")
        return pd.to_numeric(d["bvalue"], errors="coerce").to_numpy(dtype=float)

    # Preferimos columnas precomputadas (si están)
    bcol_map = {"g": "bvalue_g", "g_lin_max": "bvalue_g_lin_max", "g_thorsten": "bvalue_thorsten"}
    bcol = bcol_map.get(g_type)
    if bcol and bcol in d.columns:
        return pd.to_numeric(d[bcol], errors="coerce").to_numpy(dtype=float)

    # fallback a b_from_g
    if g_type not in d.columns:
        raise ValueError(f"Falta columna '{g_type}' (y tampoco existe {bcol}).")

    if N is None or delta_ms is None or Delta_app_ms is None:
        raise ValueError("Para g_type!=bvalue necesitás N, delta_ms y Delta_app_ms (args o columnas únicas).")

    g = pd.to_numeric(d[g_type], errors="coerce").to_numpy(dtype=float)
    return b_from_g(
        g,
        N=float(N),
        gamma=float(gamma),
        delta_ms=float(delta_ms),
        delta_app_ms=float(Delta_app_ms),
        g_type=("gthorsten" if g_type == "g_thorsten" else g_type),
    )


@dataclass(frozen=True)
class FitOutputs:
    fit_params: pd.DataFrame
    fit_table: pd.DataFrame


def plot_fit_one_group_monoexp(
    df_group: pd.DataFrame,
    fit_row: dict,
    *,
    out_png: Path,
    ycol: str,
    g_type: str,
    fit_points: int,
) -> None:
    df_group = _ensure_keys_types(df_group.copy())

    if ycol not in df_group.columns:
        raise KeyError(f"No encuentro ycol='{ycol}' en df_group.")

    N = fit_row.get("N")
    delta_ms = fit_row.get("delta_ms")
    Delta_app_ms = fit_row.get("Delta_app_ms")
    gamma = 267.5221900

    y = pd.to_numeric(df_group[ycol], errors="coerce").to_numpy(dtype=float)
    b = _b_from_mode(
        df_group,
        g_type=g_type,
        gamma=gamma,
        N=None if pd.isna(N) else float(N),
        delta_ms=None if pd.isna(delta_ms) else float(delta_ms),
        Delta_app_ms=None if pd.isna(Delta_app_ms) else float(Delta_app_ms),
    )

    m = np.isfinite(y) & np.isfinite(b)
    y = y[m]
    b = b[m]
    if len(b) == 0:
        raise ValueError("No hay puntos válidos para plotear.")

    bmax = float(np.nanmax(b)) if np.any(np.isfinite(b)) else 0.0
    b_dense = np.linspace(0.0, bmax, 250)

    ys = None
    label = "monoexp"
    if bool(fit_row.get("ok", True)):
        M0 = float(fit_row["M0"])
        D0 = float(fit_row["D0_mm2_s"])
        ys = monoexp(b_dense, M0, D0)
        label = f"monoexp: M0={M0:.3g}, D0={D0:.3g} mm²/s"

    start_fit_figure()
    plot_fit_data(b, y, label="data")
    highlight_fit_points(b[: int(fit_points)], y[: int(fit_points)], label=f"fit first {int(fit_points)}")
    if ys is not None:
        plot_fit_curve(b_dense, ys, label=label)

    roi = fit_row.get("roi", "roi")
    direction = fit_row.get("direction", "direction")
    td = fit_row.get("td_ms")
    td_txt = f"{float(td):.1f}" if td is not None and not pd.isna(td) else "NA"
    N_txt = f"{int(round(float(N)))}" if N is not None and not pd.isna(N) else "NA"

    plt.yscale("log")
    finish_fit_figure(
        title=f"OGSE signal fit | ROI={roi} | direction={direction} | $t_d$={td_txt} ms | N={N_txt}",
        xlabel="bvalue (s/mm$^2$)",
        ylabel=ycol,
        out_png=out_png,
    )


def fit_signal_vs_bval_long(
    df: pd.DataFrame,
    *,
    dirs: Optional[Sequence[str]] = None,
    roi: str = "ALL",
    rois: Optional[Sequence[str]] = None,
    ycol: str = "value_norm",
    fit_points: int = 6,
    g_type: str = "bvalue",
    gamma: float = 267.5221900,
    N: Optional[float] = None,
    delta_ms: Optional[float] = None,
    Delta_app_ms: Optional[float] = None,
    td_ms: float,
    D0_init: float = 0.0023,
    free_M0: bool = False,
    fix_M0: float = 1.0,
    outdir_plots: Optional[Path] = None,
    title_prefix: str = "",
    stat_keep: str = "avg",
) -> FitOutputs:
    dfa = df.copy()
    _require_no_axis(dfa)

    for c in ["direction", "roi", "b_step"]:
        if c not in dfa.columns:
            raise ValueError(f"Falta columna requerida '{c}'. Columns={list(dfa.columns)}")

    if ycol not in {"value", "value_norm"}:
        raise ValueError("ycol debe ser 'value' o 'value_norm'. Este pipeline no admite nombres legacy.")
    if ycol not in dfa.columns:
        raise ValueError(f"Falta ycol='{ycol}'. Columns={list(dfa.columns)}")

    dfa = _ensure_keys_types(dfa)

    if "stat" in dfa.columns and stat_keep is not None and str(stat_keep).upper() != "ALL":
        dfa = dfa[dfa["stat"] == str(stat_keep)].copy()
        if dfa.empty:
            raise ValueError(f"No quedaron filas con stat == '{stat_keep}'.")

    if dirs is not None:
        dfa = dfa[dfa["direction"].isin([str(x) for x in dirs])].copy()
    rois_keep = _normalize_requested_rois(roi=roi, rois=rois)
    if rois_keep is not None:
        dfa = dfa[dfa["roi"].isin(rois_keep)].copy()
    if dfa.empty:
        dirs_avail = sorted(df["direction"].astype(str).dropna().unique().tolist()) if "direction" in df.columns else []
        rois_avail = sorted(df["roi"].astype(str).dropna().unique().tolist()) if "roi" in df.columns else []
        raise ValueError(
            "No quedaron filas luego de filtrar direction/roi. "
            f"Direcciones disponibles={dirs_avail}. ROIs disponibles={rois_avail}."
        )

    if outdir_plots is not None:
        outdir_plots.mkdir(parents=True, exist_ok=True)

    results = []
    points = []

    for (dir_val, roi_val), d in dfa.groupby(["direction", "roi"], sort=False):
        d = d.sort_values("b_step", kind="stable").copy()

        y = pd.to_numeric(d[ycol], errors="coerce").to_numpy(dtype=float)
        b = _b_from_mode(d, g_type=g_type, gamma=gamma, N=N, delta_ms=delta_ms, Delta_app_ms=Delta_app_ms)

        k = min(int(fit_points), len(b))
        b_fit = b[:k].copy()
        y_fit = y[:k].copy()

        m = np.isfinite(b_fit) & np.isfinite(y_fit) & (y_fit > 0)
        b_fit, y_fit = b_fit[m], y_fit[m]

        if len(b_fit) < 3:
            results.append(
                dict(
                    roi=str(roi_val), direction=str(dir_val), stat=str(stat_keep),
                    model="monoexp", ycol=str(ycol), g_type=str(g_type), fit_points=int(fit_points),
                    n_points=int(len(b)), n_fit=int(len(b_fit)), td_ms=float(td_ms), ok=False, msg="Muy pocos puntos válidos."
                )
            )
            continue

        if free_M0:
            p0 = [1.0, D0_init]
            bounds = ([0.0, D0_init / 10], [100.0, 2 * D0_init])
            popt, pcov = curve_fit(monoexp, b_fit, y_fit, p0=p0, bounds=bounds, maxfev=40000)
            M0_hat, D0_hat = float(popt[0]), float(popt[1])
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.array([np.nan, np.nan])
            M0_err, D0_err = float(perr[0]), float(perr[1])
            method = "curve_fit(M0,D0)"
        else:
            f = lambda bb, D0: monoexp_M0fixed(bb, D0, M0=float(fix_M0))
            p0 = [D0_init]
            bounds = ([D0_init / 10], [2 * D0_init])
            popt, pcov = curve_fit(f, b_fit, y_fit, p0=p0, bounds=bounds, maxfev=40000)
            M0_hat, D0_hat = float(fix_M0), float(popt[0])
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else np.array([np.nan])
            M0_err, D0_err = np.nan, float(perr[0])
            method = "curve_fit(D0) M0_fixed"

        yhat = monoexp(b_fit, M0_hat, D0_hat)
        rmse = _rmse(y_fit, yhat)
        chi2 = _chi2(y_fit, yhat)

        results.append(
            dict(
                roi=str(roi_val), direction=str(dir_val), stat=str(stat_keep),
                model="monoexp", ycol=str(ycol), g_type=str(g_type), fit_points=int(fit_points),
                n_points=int(len(b)), n_fit=int(len(b_fit)), td_ms=float(td_ms),
                N=float(N) if N is not None else np.nan,
                delta_ms=float(delta_ms) if delta_ms is not None else np.nan,
                Delta_app_ms=float(Delta_app_ms) if Delta_app_ms is not None else np.nan,
                M0=float(M0_hat), M0_err=float(M0_err) if np.isfinite(M0_err) else np.nan,
                D0_mm2_s=float(D0_hat), D0_err_mm2_s=float(D0_err) if np.isfinite(D0_err) else np.nan,
                D0_m2_ms=float(D0_hat) * 1e-9,
                D0_err_m2_ms=float(D0_err) * 1e-9 if np.isfinite(D0_err) else np.nan,
                rmse=float(rmse), chi2=float(chi2),
                method=method, ok=True, msg="",
            )
        )

        used = np.zeros(len(b), dtype=bool); used[:k] = True
        for bs, bi, yi, uu in zip(d["b_step"].to_numpy(), b, y, used):
            points.append(
                dict(
                    roi=str(roi_val), direction=str(dir_val), stat=str(stat_keep),
                    ycol=str(ycol), g_type=str(g_type), fit_points=int(fit_points), td_ms=float(td_ms),
                    b_step=int(bs), bvalue_used=float(bi) if np.isfinite(bi) else np.nan,
                    y=float(yi) if np.isfinite(yi) else np.nan, used_for_fit=bool(uu),
                )
            )

        if outdir_plots is not None:
            out_png = outdir_plots / f"{roi_val}.monoexp.{g_type}.{ycol}.direction_{dir_val}.png"
            plot_fit_one_group_monoexp(
                d,
                results[-1],
                out_png=out_png,
                ycol=ycol,
                g_type=g_type,
                fit_points=int(fit_points),
            )

    return FitOutputs(fit_params=pd.DataFrame(results), fit_table=pd.DataFrame(points))


def run_fit_from_parquet(
    parquet_path: str | Path,
    *,
    dirs: Optional[Sequence[str]] = None,
    roi: str = "ALL",
    rois: Optional[Sequence[str]] = None,
    ycol: str = "value_norm",
    g_type: str = "bvalue",
    fit_points: int = 6,
    free_M0: bool = False,
    fix_M0: float = 1.0,
    D0_init: float = 0.0023,
    gamma: float = 267.5221900,
    N: Optional[float] = None,
    delta_ms: Optional[float] = None,
    Delta_app_ms: Optional[float] = None,
    td_ms: Optional[float] = None,
    stat_keep: str = "avg",
    out_root: str | Path = "ogse_experiments/fits/fit-monoexp_ogse-signal",
) -> Tuple[FitOutputs, Path]:
    p = Path(parquet_path)
    df = pd.read_parquet(p) if p.suffix.lower() not in [".xlsx", ".xls"] else pd.read_excel(p, sheet_name=0)

    _require_no_axis(df)
    df = _ensure_keys_types(df)

    exp_id = infer_exp_id(p)

    # --- layout final: out_root/<SHEET>/<exp_id>/{tables,plots} ---
    sheet = _unique_str(df, "sheet") or exp_id.split("_")[0]

    exp_dir = Path(out_root) / sheet / exp_id
    tables_dir = exp_dir
    plots_dir = exp_dir 

    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # correr fit (plots al directorio correcto)
    outs = fit_signal_vs_bval_long(
        df,
        dirs=dirs,
        roi=roi,
        rois=rois,
        ycol=ycol,
        fit_points=fit_points,
        g_type=g_type,
        gamma=gamma,
        N=N,
        delta_ms=delta_ms,
        Delta_app_ms=Delta_app_ms,
        td_ms=td_ms,
        D0_init=D0_init,
        free_M0=free_M0,
        fix_M0=fix_M0,
        outdir_plots=plots_dir,
        title_prefix=f"{exp_id} | ",
        stat_keep=stat_keep,
    )

    # agregar metadata escalar útil
    fp = outs.fit_params.copy()
    if not fp.empty:
        fp["max_dur_ms"] = _unique_float_any(df, ["max_dur_ms"], required=False, name="max_dur_ms")
        fp["tm_ms"] = _unique_float_any(df, ["tm_ms"], required=False, name="tm_ms")
        fp["td_ms"] = td_ms if td_ms is not None else _unique_float_any(df, ["td_ms"], required=False, name="td_ms")
        fp = standardize_fit_params(fp, fit_kind="monoexp", source_file=p.name)

    # guardar tablas (nombres estables)
    out_params_parquet = tables_dir / "fit_params.parquet"
    out_params_xlsx    = tables_dir / "fit_params.xlsx"
    out_points_parquet = tables_dir / "fit_points.parquet"
    out_points_xlsx    = tables_dir / "fit_points.xlsx"

    fp.to_parquet(out_params_parquet, index=False)
    fp.to_excel(out_params_xlsx, index=False)
    outs.fit_table.to_parquet(out_points_parquet, index=False)
    outs.fit_table.to_excel(out_points_xlsx, index=False)

    return outs, exp_dir
