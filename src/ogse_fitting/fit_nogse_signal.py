from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fitting.core import chi2 as fit_chi2
from fitting.core import fit_least_squares, format_value_error
from fitting.core import r2_score, rmse as fit_rmse
from nogse_models.nogse_model_fitting import M_nogse_free
from tools.fit_params_schema import standardize_fit_params
from data_processing.io import write_table_outputs
from tools.value_formatting import scalar_or_compact_column


@dataclass(frozen=True)
class SignalModelSpec:
    name: str
    x_model_ms: Callable[[float], float]
    evaluator: Callable[[float, np.ndarray, int, float, float, float], np.ndarray]


@dataclass(frozen=True)
class FitOutputs:
    fit_params: pd.DataFrame


@dataclass(frozen=True)
class BuiltModel:
    model: Callable[..., np.ndarray]
    x_model_ms: float
    param_names: list[str]
    p0: list[float]
    bounds: tuple[list[float], list[float]]


SIGNAL_MODELS: dict[str, SignalModelSpec] = {
    "free_cpmg": SignalModelSpec(
        name="free_cpmg",
        x_model_ms=lambda tn_ms: 0.5 * float(tn_ms),
        evaluator=M_nogse_free,
    ),
    "free_hahn": SignalModelSpec(
        name="free_hahn",
        x_model_ms=lambda _tn_ms: 0.0,
        evaluator=M_nogse_free,
    ),
}
VALID_MODELS = set(SIGNAL_MODELS)


def analysis_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith(".long"):
        stem = stem[: -len(".long")]
    return stem


def split_all_or_values(values: Sequence[str] | None) -> list[str] | None:
    if values is None:
        return None
    values = [str(v) for v in values]
    if len(values) == 1 and values[0].upper() == "ALL":
        return None
    return values


def _unique_scalar(df: pd.DataFrame, col: str, *, required: bool = False):
    if col not in df.columns:
        if required:
            raise ValueError(f"Missing required column {col!r}.")
        return None
    values = pd.Series(df[col]).dropna().unique().tolist()
    if len(values) == 0:
        if required:
            raise ValueError(f"Column {col!r} is empty.")
        return None
    if len(values) > 1:
        raise ValueError(f"Column {col!r} is not unique inside the group: {values}")
    return values[0]


def _resolve_tn_ms(df: pd.DataFrame) -> float:
    for col in ("TN", "td_ms"):
        value = _unique_scalar(df, col, required=False)
        if value is None:
            continue
        value = float(value)
        if np.isfinite(value):
            return value
    raise ValueError("Could not infer TN/td_ms from the signal table.")


def _resolve_sigma(avg_df: pd.DataFrame, std_df: pd.DataFrame | None, *, xcol: str, ycol: str) -> np.ndarray | None:
    if std_df is None or std_df.empty:
        return None

    err = std_df.copy()
    err[xcol] = pd.to_numeric(err[xcol], errors="coerce")
    err["value"] = pd.to_numeric(err["value"], errors="coerce")
    err = err.dropna(subset=[xcol, "value"]).sort_values(xcol)
    if err.empty:
        return None

    sigma = err["value"].to_numpy(dtype=float)
    if ycol == "value_norm":
        s0 = pd.to_numeric(avg_df["S0"], errors="coerce").to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma = sigma / s0

    if sigma.shape[0] != avg_df.shape[0]:
        return None
    if not np.isfinite(sigma).any():
        return None
    sigma[~np.isfinite(sigma)] = np.nanmax(sigma[np.isfinite(sigma)])
    sigma[sigma <= 0] = np.nanmax(sigma[sigma > 0]) if np.any(sigma > 0) else 1.0
    return sigma


def validate_bounds(name: str, bounds: Sequence[float]) -> tuple[float, float]:
    lower, upper = float(bounds[0]), float(bounds[1])
    if not lower < upper:
        raise ValueError(f"{name} lower bound must be smaller than upper bound: {lower}, {upper}")
    return lower, upper


def _clamp_p0(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    return float(min(max(float(value), lower), upper))


def validate_fixed_value(name: str, value: float | None, bounds: tuple[float, float]) -> None:
    if value is None:
        return
    lower, upper = bounds
    if not lower <= float(value) <= upper:
        raise ValueError(f"{name} fixed value {value} is outside bounds [{lower}, {upper}].")


def validate_log_bounds(name: str, bounds: tuple[float, float]) -> None:
    lower, _upper = bounds
    if lower <= 0:
        raise ValueError(f"{name} lower bound must be positive when fitting in log-space: {lower}")


def _build_model(
    model_name: str,
    *,
    tn_ms: float,
    n_value: int,
    fix_m0: float | None,
    fix_d0: float | None,
    m0_bounds: tuple[float, float],
    d0_bounds: tuple[float, float],
) -> BuiltModel:
    if model_name not in SIGNAL_MODELS:
        raise ValueError(f"Unsupported NOGSE signal model {model_name!r}. Allowed values: {sorted(SIGNAL_MODELS)}.")

    spec = SIGNAL_MODELS[model_name]
    x_ms = float(spec.x_model_ms(float(tn_ms)))
    m0_lo, m0_hi = m0_bounds
    d0_lo, d0_hi = d0_bounds

    param_names: list[str] = []
    p0: list[float] = []
    lower: list[float] = []
    upper: list[float] = []

    if fix_m0 is None:
        param_names.append("M0")
        p0.append(_clamp_p0(1.0, m0_bounds))
        lower.append(m0_lo)
        upper.append(m0_hi)
    if fix_d0 is None:
        param_names.append("D0_m2_ms")
        p0.append(_clamp_p0(2.3e-12, d0_bounds))
        lower.append(d0_lo)
        upper.append(d0_hi)

    def model(g: np.ndarray, *values: float) -> np.ndarray:
        fitted = dict(zip(param_names, values))
        m0 = float(fix_m0) if fix_m0 is not None else float(fitted["M0"])
        d0 = float(fix_d0) if fix_d0 is not None else float(fitted["D0_m2_ms"])
        return spec.evaluator(float(tn_ms), g, int(n_value), x_ms, m0, d0)

    return BuiltModel(model=model, x_model_ms=x_ms, param_names=param_names, p0=p0, bounds=(lower, upper))


def fit_one_group(
    avg_df: pd.DataFrame,
    std_df: pd.DataFrame | None,
    *,
    model_name: str,
    xcol: str,
    ycol: str,
    fix_m0: float | None,
    fix_d0: float | None,
    m0_bounds: tuple[float, float],
    d0_bounds: tuple[float, float],
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    data = avg_df.copy()
    data[xcol] = pd.to_numeric(data[xcol], errors="coerce")
    data[ycol] = pd.to_numeric(data[ycol], errors="coerce")
    data = data.dropna(subset=[xcol, ycol]).sort_values(xcol)
    if data.empty:
        raise ValueError("No valid points remain after numeric filtering.")

    tn_ms = _resolve_tn_ms(data)
    n_value = int(round(float(_unique_scalar(data, "N", required=True))))
    built = _build_model(
        model_name,
        tn_ms=tn_ms,
        n_value=n_value,
        fix_m0=fix_m0,
        fix_d0=fix_d0,
        m0_bounds=m0_bounds,
        d0_bounds=d0_bounds,
    )

    x = data[xcol].to_numpy(dtype=float)
    y = data[ycol].to_numpy(dtype=float)
    sigma = _resolve_sigma(data, std_df, xcol=xcol, ycol=ycol)

    p0 = list(built.p0)
    if "M0" in built.param_names:
        p0[built.param_names.index("M0")] = _clamp_p0(float(np.nanmax(y)), m0_bounds)

    if built.param_names:
        fit = fit_least_squares(
            built.model,
            x,
            y,
            param_names=built.param_names,
            p0=p0,
            bounds=built.bounds,
            log_params=["D0_m2_ms"] if "D0_m2_ms" in built.param_names else [],
            sigma=sigma,
            max_nfev=200000,
            method="least_squares_logD" if "D0_m2_ms" in built.param_names else "least_squares",
        )
        yhat = fit.yhat
        values = fit.values
        errors = fit.errors
        method = fit.method
        rmse = fit.rmse
        chi2 = fit.chi2
        r2 = fit.r2
        popt = [values[name] for name in built.param_names]
    else:
        yhat = built.model(x)
        values = {}
        errors = {}
        method = "fixed"
        rmse = fit_rmse(y, yhat)
        chi2 = fit_chi2(y, yhat)
        r2 = r2_score(y, yhat)
        popt = []

    fit_row: dict[str, object] = {
        "model": model_name,
        "xcol": xcol,
        "ycol": ycol,
        "TN": tn_ms,
        "N": n_value,
        "x_model_ms": built.x_model_ms,
        "x_min": float(np.nanmin(x)),
        "x_max": float(np.nanmax(x)),
        "rmse": float(rmse),
        "r2": float(r2),
        "chi2": float(chi2),
        "method": method,
        "n_points": int(len(x)),
        "n_fit": int(len(x)),
        "M0_bound_min": float(m0_bounds[0]),
        "M0_bound_max": float(m0_bounds[1]),
        "D0_bound_min": float(d0_bounds[0]),
        "D0_bound_max": float(d0_bounds[1]),
    }
    if fix_m0 is not None:
        fit_row["M0"] = float(fix_m0)
        fit_row["M0_err"] = np.nan
    if fix_d0 is not None:
        fit_row["D0_m2_ms"] = float(fix_d0)
        fit_row["D0_err_m2_ms"] = np.nan
    fit_row.update(values)
    fit_row.update(errors)

    xfit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 400)
    yfit = built.model(xfit, *popt) if built.param_names else built.model(xfit)
    return fit_row, x, y, np.column_stack([xfit, yfit])


def plot_fit_one_group(
    avg_df: pd.DataFrame,
    std_df: pd.DataFrame | None,
    *,
    xcol: str,
    ycol: str,
    fit_row: dict[str, object],
    x_data: np.ndarray,
    y_data: np.ndarray,
    fit_curve: np.ndarray,
    out_png: Path,
    title: str,
) -> None:
    sigma = _resolve_sigma(avg_df, std_df, xcol=xcol, ycol=ycol)

    plt.figure(figsize=(8, 6))
    if sigma is not None:
        plt.errorbar(x_data, y_data, yerr=sigma, fmt="o", markersize=6, capsize=3, label="signal")
    else:
        plt.plot(x_data, y_data, "o", markersize=6, label="signal")

    plt.plot(fit_curve[:, 0], fit_curve[:, 1], "-", linewidth=2, label=fit_row["model"])
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)

    d0_m2_ms = float(fit_row.get("D0_m2_ms", np.nan))
    d0_err_m2_ms = fit_row.get("D0_err_m2_ms", None)
    # d0_mm2_s = d0_m2_ms * 1e9 if np.isfinite(d0_m2_ms) else np.nan
    # d0_err_mm2_s = float(d0_err_m2_ms) * 1e9 if d0_err_m2_ms is not None and np.isfinite(float(d0_err_m2_ms)) else np.nan

    text_lines = [
        f"model={fit_row['model']}",
        f"M0={format_value_error(fit_row.get('M0', np.nan), fit_row.get('M0_err', None))}",
        f"D0={format_value_error(d0_m2_ms, d0_err_m2_ms, unit='m2/ms')}",
        # f"D0={format_value_error(d0_mm2_s, d0_err_mm2_s, unit='mm2/s')}",
        f"rmse={fit_row['rmse']:.4g}",
        f"R2={fit_row['r2']:.4g}",
    ]
    plt.text(
        0.02,
        0.98,
        "\n".join(text_lines),
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def fit_nogse_signal_long(
    df: pd.DataFrame,
    *,
    model: str,
    xcol: str = "g",
    ycol: str = "value_norm",
    stat_keep: str = "avg",
    rois: Sequence[str] | None = None,
    directions: Sequence[str] | None = None,
    fix_m0: float | None = None,
    fix_d0: float | None = None,
    m0_bounds: tuple[float, float] = (0.0, np.inf),
    d0_bounds: tuple[float, float] = (1e-16, np.inf),
    source_file: str | None = None,
    analysis_id: str | None = None,
    outdir_plots: Path | None = None,
) -> FitOutputs:
    if model not in SIGNAL_MODELS:
        raise ValueError(f"Unsupported NOGSE signal model {model!r}. Allowed values: {sorted(SIGNAL_MODELS)}.")

    avg_df = df[df["stat"].astype(str) == str(stat_keep)].copy()
    if avg_df.empty:
        raise ValueError(f"No rows found for stat={stat_keep!r}.")

    std_df = df[df["stat"].astype(str) == "std"].copy()

    rois_keep = split_all_or_values(rois)
    directions_keep = split_all_or_values(directions)
    if rois_keep is not None:
        avg_df = avg_df[avg_df["roi"].astype(str).isin(rois_keep)].copy()
        std_df = std_df[std_df["roi"].astype(str).isin(rois_keep)].copy()
    if directions_keep is not None:
        avg_df = avg_df[avg_df["direction"].astype(str).isin(directions_keep)].copy()
        std_df = std_df[std_df["direction"].astype(str).isin(directions_keep)].copy()

    if avg_df.empty:
        raise ValueError("No data remains after ROI/direction filtering.")

    fit_rows: list[dict[str, object]] = []
    for (roi, direction), group in avg_df.groupby(["roi", "direction"], sort=False):
        std_group = None
        if not std_df.empty:
            std_group = std_df[
                (std_df["roi"].astype(str) == str(roi))
                & (std_df["direction"].astype(str) == str(direction))
            ].copy()

        fit_row, x_data, y_data, fit_curve = fit_one_group(
            group,
            std_group,
            model_name=model,
            xcol=xcol,
            ycol=ycol,
            fix_m0=fix_m0,
            fix_d0=fix_d0,
            m0_bounds=m0_bounds,
            d0_bounds=d0_bounds,
        )

        for col in [
            "subj",
            "sheet",
            "protocol",
            "group",
            "type",
            "TN",
            "x",
            "y",
            "Hz",
            "N",
            "TE",
            "TR",
            "delta_ms",
            "Delta_app_ms",
            "td_ms",
            "tm_ms",
            "max_dur_ms",
        ]:
            if col in group.columns:
                fit_row[col] = _unique_scalar(group, col, required=False)

        if "sequence" in group.columns:
            fit_row["sequence"] = scalar_or_compact_column(group, "sequence", required=False)

        if "G" in group.columns:
            g_values = pd.to_numeric(group["G"], errors="coerce").dropna().to_numpy(dtype=float)
            if g_values.size > 0:
                fit_row["G_min"] = float(np.nanmin(g_values))
                fit_row["G_max"] = float(np.nanmax(g_values))

        fit_row["roi"] = str(roi)
        fit_row["direction"] = str(direction)
        fit_row["stat"] = str(stat_keep)
        fit_row["source_file"] = str(source_file or "")
        fit_row["analysis_id"] = str(analysis_id or "")
        fit_rows.append(fit_row)

        if outdir_plots is not None:
            signal_type = _unique_scalar(group, "type", required=False)
            title = f"{analysis_id} | roi={roi} | direction={direction} | type={signal_type} | model={model}"
            out_png = outdir_plots / f"{analysis_id}.roi-{roi}.dir-{direction}.{model}.png"
            plot_fit_one_group(
                group,
                std_group,
                xcol=xcol,
                ycol=ycol,
                fit_row=fit_row,
                x_data=x_data,
                y_data=y_data,
                fit_curve=fit_curve,
                out_png=out_png,
                title=title,
            )
            print("Saved:", out_png)

    fit_df = pd.DataFrame(fit_rows)
    if not fit_df.empty:
        fit_df = standardize_fit_params(fit_df, fit_kind="nogse_signal", source_file=source_file)
    return FitOutputs(fit_params=fit_df)


def run_fit_from_parquet(
    parquet_path: str | Path,
    *,
    model: str,
    out_root: str | Path,
    xcol: str = "g",
    ycol: str = "value_norm",
    stat_keep: str = "avg",
    rois: Sequence[str] | None = None,
    directions: Sequence[str] | None = None,
    fix_m0: float | None = None,
    fix_d0: float | None = None,
    m0_bounds: tuple[float, float] = (0.0, np.inf),
    d0_bounds: tuple[float, float] = (1e-16, np.inf),
) -> tuple[FitOutputs, Path]:
    p = Path(parquet_path)
    df = pd.read_parquet(p)
    analysis_id = analysis_id_from_path(p)
    out_dir = Path(out_root) / model / analysis_id
    out_dir.mkdir(parents=True, exist_ok=True)

    outs = fit_nogse_signal_long(
        df,
        model=model,
        xcol=xcol,
        ycol=ycol,
        stat_keep=stat_keep,
        rois=rois,
        directions=directions,
        fix_m0=fix_m0,
        fix_d0=fix_d0,
        m0_bounds=m0_bounds,
        d0_bounds=d0_bounds,
        source_file=p.name,
        analysis_id=analysis_id,
        outdir_plots=out_dir,
    )

    out_parquet = out_dir / "fit_params.parquet"
    write_table_outputs(
        outs.fit_params,
        out_parquet,
        xlsx_path=out_parquet.with_suffix(".xlsx"),
        csv_path=out_dir / "fit_params.csv",
    )
    print("Saved fit table:", out_parquet)
    return outs, out_dir
