from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

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


def _build_model(model_name: str, *, tn_ms: float, n_value: int, fix_m0: float | None):
    x_ms = 0.5 * tn_ms if model_name == "free_cpmg" else 0.0

    if fix_m0 is None:
        def model(g, m0, d0):
            return M_nogse_free(tn_ms, g, n_value, x_ms, m0, d0)

        param_names = ["M0", "D0_m2_ms"]
        p0 = [1.0, 2.3e-12]
        bounds = ([0.0, 0.0], [np.inf, np.inf])
    else:
        def model(g, d0):
            return M_nogse_free(tn_ms, g, n_value, x_ms, fix_m0, d0)

        param_names = ["D0_m2_ms"]
        p0 = [2.3e-12]
        bounds = ([0.0], [np.inf])

    return model, x_ms, param_names, p0, bounds


def _fit_one_group(
    avg_df: pd.DataFrame,
    std_df: pd.DataFrame | None,
    *,
    model_name: str,
    xcol: str,
    ycol: str,
    fix_m0: float | None,
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
    )

    x = data[xcol].to_numpy(dtype=float)
    y = data[ycol].to_numpy(dtype=float)
    sigma = _resolve_sigma(data, std_df, xcol=xcol, ycol=ycol)

    if fix_m0 is None:
        p0[0] = float(np.nanmax(y))
    elif ycol == "value_norm":
        fix_m0 = 1.0

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
        "n_points": int(len(x)),
    }
    if fix_m0 is not None:
        fit_row["M0"] = float(fix_m0)
    for name, value in zip(param_names, popt):
        fit_row[name] = float(value)

    xfit = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 400)
    yfit = model(xfit, *popt)
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
