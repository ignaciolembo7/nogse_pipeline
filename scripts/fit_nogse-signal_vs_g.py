from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares

from nogse_models.nogse_model_fitting import M_nogse_free


VALID_MODELS = {"free_cpmg", "free_hahn"}


def _analysis_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith(".long"):
        stem = stem[: -len(".long")]
    return stem


def _split_all_or_values(values: list[str] | None) -> list[str] | None:
    if values is None:
        return None
    if len(values) == 1 and str(values[0]).upper() == "ALL":
        return None
    return [str(v) for v in values]


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


def _validate_bounds(name: str, bounds: list[float] | tuple[float, float]) -> tuple[float, float]:
    lower, upper = float(bounds[0]), float(bounds[1])
    if not lower < upper:
        raise ValueError(f"{name} lower bound must be smaller than upper bound: {lower}, {upper}")
    return lower, upper


def _clamp_p0(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    return float(min(max(float(value), lower), upper))


def _validate_fixed_value(name: str, value: float | None, bounds: tuple[float, float]) -> None:
    if value is None:
        return
    lower, upper = bounds
    if not lower <= float(value) <= upper:
        raise ValueError(f"{name} fixed value {value} is outside bounds [{lower}, {upper}].")


def _validate_log_bounds(name: str, bounds: tuple[float, float]) -> None:
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
):
    x_ms = 0.5 * tn_ms if model_name == "free_cpmg" else 0.0
    m0_lo, m0_hi = m0_bounds
    d0_lo, d0_hi = d0_bounds

    if fix_m0 is None and fix_d0 is None:
        def model(g, m0, d0):
            return M_nogse_free(tn_ms, g, n_value, x_ms, m0, d0)

        param_names = ["M0", "D0_m2_ms"]
        p0 = [_clamp_p0(1.0, m0_bounds), _clamp_p0(2.3e-12, d0_bounds)]
        bounds = ([m0_lo, d0_lo], [m0_hi, d0_hi])
    elif fix_m0 is not None and fix_d0 is None:
        def model(g, d0):
            return M_nogse_free(tn_ms, g, n_value, x_ms, fix_m0, d0)

        param_names = ["D0_m2_ms"]
        p0 = [_clamp_p0(2.3e-12, d0_bounds)]
        bounds = ([d0_lo], [d0_hi])
    elif fix_m0 is None and fix_d0 is not None:
        def model(g, m0):
            return M_nogse_free(tn_ms, g, n_value, x_ms, m0, fix_d0)

        param_names = ["M0"]
        p0 = [_clamp_p0(1.0, m0_bounds)]
        bounds = ([m0_lo], [m0_hi])
    else:
        def model(g):
            return M_nogse_free(tn_ms, g, n_value, x_ms, fix_m0, fix_d0)

        param_names = []
        p0 = []
        bounds = ([], [])

    return model, x_ms, param_names, p0, bounds


def _fit_one_group(
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
    model, x_ms, param_names, p0, bounds = _build_model(
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

    if "M0" in param_names:
        p0[param_names.index("M0")] = _clamp_p0(float(np.nanmax(y)), m0_bounds)

    if param_names:
        if "D0_m2_ms" in param_names:
            opt_p0: list[float] = []
            opt_lower: list[float] = []
            opt_upper: list[float] = []
            for name, value, lower, upper in zip(param_names, p0, bounds[0], bounds[1]):
                if name == "D0_m2_ms":
                    opt_p0.append(float(np.log(value)))
                    opt_lower.append(float(np.log(lower)))
                    opt_upper.append(float(np.log(upper)))
                else:
                    opt_p0.append(float(value))
                    opt_lower.append(float(lower))
                    opt_upper.append(float(upper))

            def _unpack_opt(values: np.ndarray) -> list[float]:
                out: list[float] = []
                for name, value in zip(param_names, values):
                    out.append(float(np.exp(value)) if name == "D0_m2_ms" else float(value))
                return out

            def _residual(values: np.ndarray) -> np.ndarray:
                yhat_local = model(x, *_unpack_opt(values))
                residual = yhat_local - y
                if sigma is not None:
                    residual = residual / sigma
                return residual

            opt = least_squares(
                _residual,
                np.asarray(opt_p0, dtype=float),
                bounds=(np.asarray(opt_lower, dtype=float), np.asarray(opt_upper, dtype=float)),
                x_scale="jac",
                max_nfev=200000,
            )
            if not opt.success:
                raise RuntimeError(f"least_squares failed: {opt.message}")
            popt = np.asarray(_unpack_opt(opt.x), dtype=float)
            yhat = model(x, *popt)
            fit_method = "least_squares_logD"
        else:
            popt, _pcov = curve_fit(
                model,
                x,
                y,
                p0=p0,
                bounds=bounds,
                sigma=sigma,
                absolute_sigma=False,
                maxfev=200000,
            )
            yhat = model(x, *popt)
            fit_method = "scipy_curve_fit"
    else:
        popt = np.array([], dtype=float)
        yhat = model(x)
        fit_method = "fixed"
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    fit_row: dict[str, object] = {
        "model": model_name,
        "xcol": xcol,
        "ycol": ycol,
        "TN": tn_ms,
        "N": n_value,
        "x_model_ms": x_ms,
        "x_min": float(np.nanmin(x)),
        "x_max": float(np.nanmax(x)),
        "rmse": rmse,
        "r2": r2,
        "method": fit_method,
        "n_points": int(len(x)),
        "M0_bound_min": float(m0_bounds[0]),
        "M0_bound_max": float(m0_bounds[1]),
        "D0_bound_min": float(d0_bounds[0]),
        "D0_bound_max": float(d0_bounds[1]),
    }
    if fix_m0 is not None:
        fit_row["M0"] = float(fix_m0)
    if fix_d0 is not None:
        fit_row["D0_m2_ms"] = float(fix_d0)
    for name, value in zip(param_names, popt):
        fit_row[name] = float(value)

    xfit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 400)
    yfit = model(xfit, *popt) if param_names else model(xfit)
    return fit_row, x, y, np.column_stack([xfit, yfit])


def _plot_fit(
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
        plt.errorbar(
            x_data,
            y_data,
            yerr=sigma,
            fmt="o",
            markersize=6,
            capsize=3,
            label="signal",
        )
    else:
        plt.plot(x_data, y_data, "o", markersize=6, label="signal")

    plt.plot(fit_curve[:, 0], fit_curve[:, 1], "-", linewidth=2, label=fit_row["model"])
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)

    text_lines = [
        f"model={fit_row['model']}",
        f"M0={fit_row.get('M0', np.nan):.6g}",
        f"D0={fit_row.get('D0_m2_ms', np.nan):.6g}",
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("signal_parquet", type=Path)
    ap.add_argument("--model", required=True, choices=sorted(VALID_MODELS))
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--xcol", default="g")
    ap.add_argument("--ycol", default="value_norm")
    ap.add_argument("--stat", default="avg")
    ap.add_argument("--rois", nargs="*", default=None)
    ap.add_argument("--directions", nargs="*", default=None)
    m0_group = ap.add_mutually_exclusive_group()
    m0_group.add_argument("--fix_M0", type=float, default=None)
    m0_group.add_argument("--free_M0", action="store_true")
    d0_group = ap.add_mutually_exclusive_group()
    d0_group.add_argument("--fix_D0", type=float, default=None)
    d0_group.add_argument("--free_D0", action="store_true")
    ap.add_argument("--M0_bounds", "--M0-bounds", nargs=2, type=float, default=(0.0, np.inf), metavar=("MIN", "MAX"))
    ap.add_argument("--D0_bounds", "--D0-bounds", nargs=2, type=float, default=(1e-16, np.inf), metavar=("MIN", "MAX"))
    args = ap.parse_args()

    rois = _split_all_or_values(args.rois)
    directions = _split_all_or_values(args.directions)

    df = pd.read_parquet(args.signal_parquet)
    analysis_id = _analysis_id_from_path(args.signal_parquet)

    avg_df = df[df["stat"].astype(str) == str(args.stat)].copy()
    if avg_df.empty:
        raise SystemExit(f"No rows found for stat={args.stat!r} in {args.signal_parquet}.")

    std_df = df[df["stat"].astype(str) == "std"].copy()

    if rois is not None:
        avg_df = avg_df[avg_df["roi"].astype(str).isin(rois)].copy()
        std_df = std_df[std_df["roi"].astype(str).isin(rois)].copy()
    if directions is not None:
        avg_df = avg_df[avg_df["direction"].astype(str).isin(directions)].copy()
        std_df = std_df[std_df["direction"].astype(str).isin(directions)].copy()

    if avg_df.empty:
        raise SystemExit("No data remains after ROI/direction filtering.")

    fix_m0 = args.fix_M0
    if fix_m0 is None and not args.free_M0 and args.ycol == "value_norm":
        fix_m0 = 1.0
    fix_d0 = args.fix_D0
    m0_bounds = _validate_bounds("M0", args.M0_bounds)
    d0_bounds = _validate_bounds("D0", args.D0_bounds)
    _validate_fixed_value("M0", fix_m0, m0_bounds)
    _validate_fixed_value("D0", fix_d0, d0_bounds)
    if fix_d0 is None:
        _validate_log_bounds("D0", d0_bounds)

    out_dir = args.out_root / args.model / analysis_id
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_rows: list[dict[str, object]] = []
    for (roi, direction), group in avg_df.groupby(["roi", "direction"], sort=False):
        std_group = None
        if not std_df.empty:
            std_group = std_df[
                (std_df["roi"].astype(str) == str(roi))
                & (std_df["direction"].astype(str) == str(direction))
            ].copy()

        fit_row, x_data, y_data, fit_curve = _fit_one_group(
            group,
            std_group,
            model_name=args.model,
            xcol=args.xcol,
            ycol=args.ycol,
            fix_m0=fix_m0,
            fix_d0=fix_d0,
            m0_bounds=m0_bounds,
            d0_bounds=d0_bounds,
        )

        for col in [
            "subj", "sheet", "protocol", "sequence", "group", "type",
            "TN", "x", "y", "Hz", "N", "TE", "TR", "delta_ms",
            "Delta_app_ms", "td_ms", "tm_ms", "max_dur_ms",
        ]:
            if col in group.columns:
                value = _unique_scalar(group, col, required=False)
                fit_row[col] = value

        if "G" in group.columns:
            g_values = pd.to_numeric(group["G"], errors="coerce").dropna().to_numpy(dtype=float)
            if g_values.size > 0:
                fit_row["G_min"] = float(np.nanmin(g_values))
                fit_row["G_max"] = float(np.nanmax(g_values))

        fit_row["roi"] = str(roi)
        fit_row["direction"] = str(direction)
        fit_row["source_file"] = str(args.signal_parquet.name)
        fit_row["analysis_id"] = analysis_id
        fit_rows.append(fit_row)

        signal_type = _unique_scalar(group, "type", required=False)
        title = (
            f"{analysis_id} | roi={roi} | direction={direction} | "
            f"type={signal_type} | model={args.model}"
        )
        out_png = out_dir / f"{analysis_id}.roi-{roi}.dir-{direction}.{args.model}.png"
        _plot_fit(
            group,
            std_group,
            xcol=args.xcol,
            ycol=args.ycol,
            fit_row=fit_row,
            x_data=x_data,
            y_data=y_data,
            fit_curve=fit_curve,
            out_png=out_png,
            title=title,
        )
        print("Saved:", out_png)

    fit_df = pd.DataFrame(fit_rows)
    out_parquet = out_dir / "fit_params.parquet"
    fit_df.to_parquet(out_parquet, index=False)
    fit_df.to_excel(out_parquet.with_suffix(".xlsx"), index=False)
    fit_df.to_csv(out_dir / "fit_params.csv", index=False)
    print("Saved fit table:", out_parquet)


if __name__ == "__main__":
    main()
