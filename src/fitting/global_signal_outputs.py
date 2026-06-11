from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing.io import write_table_outputs
from fitting.model_registry import SignalModelSpec


def build_contrast_fit_rows(signal_rows: list[dict[str, Any]], *, model_spec: SignalModelSpec) -> list[dict[str, Any]]:
    """Collapse fitted signal sides into contrast-level fit rows when both sides exist."""
    df = pd.DataFrame(signal_rows)
    if df.empty:
        return []

    rows: list[dict[str, Any]] = []
    group_cols = ["pair_key", "roi", "direction", "stat"]
    for _, sub in df.groupby(group_cols, sort=True, dropna=False):
        side_rows = {
            int(row["contrast_side"]): row
            for _, row in sub.iterrows()
            if int(row.get("contrast_side", 0) or 0) in (1, 2)
        }
        if 1 not in side_rows or 2 not in side_rows:
            continue
        row1 = side_rows[1]
        row2 = side_rows[2]
        base = row1.to_dict()
        base["source_file"] = str(row1.get("contrast_source_file") or row1.get("source_file", ""))
        base["analysis_id"] = str(row1.get("contrast_analysis_id") or row1.get("analysis_id", ""))
        base["fit_kind"] = f"{model_spec.family}_contrast_from_global_signal_fit"
        base["model"] = model_spec.name
        base["fit_scope"] = "global_subj_roi_direction"
        base["global_params"] = row1.get("global_params", "")
        base["global_contrast_params"] = row1.get("global_contrast_params", "")
        base["local_params"] = row1.get("local_params", "")
        base["fixed_params"] = row1.get("fixed_params", "")
        base["N_1"] = row1.get("N", np.nan)
        base["N_2"] = row2.get("N", np.nan)
        base["td_ms"] = row1.get("td_ms", np.nan)
        base["td_ms_1"] = row1.get("td_ms", np.nan)
        base["td_ms_2"] = row2.get("td_ms", np.nan)
        base["source_file_1"] = row1.get("source_file", "")
        base["source_file_2"] = row2.get("source_file", "")
        base["f_corr_1"] = row1.get("f_corr", np.nan)
        base["f_corr_2"] = row2.get("f_corr", np.nan)
        base["corr_status_1"] = row1.get("corr_status", "")
        base["corr_status_2"] = row2.get("corr_status", "")
        base["xplot"] = "1"
        base["plot_xcol"] = f"{row1.get('gbase', row1.get('xcol', 'g'))}_1"
        base["n_points"] = int(row1.get("n_points", 0) or 0) + int(row2.get("n_points", 0) or 0)
        base["n_fit"] = int(row1.get("n_fit", 0) or 0) + int(row2.get("n_fit", 0) or 0)
        base["rmse_side_1"] = row1.get("rmse", np.nan)
        base["rmse_side_2"] = row2.get("rmse", np.nan)
        base["chi2_side_1"] = row1.get("chi2", np.nan)
        base["chi2_side_2"] = row2.get("chi2", np.nan)
        base["rmse"] = np.nanmean([base["rmse_side_1"], base["rmse_side_2"]])
        base["chi2"] = np.nansum([base["chi2_side_1"], base["chi2_side_2"]])
        base["ok"] = bool(row1.get("ok", False)) and bool(row2.get("ok", False))
        rows.append(base)
    return rows


def sanitize_token(value: object) -> str:
    text = str(value)
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-") or "NA"


def plot_curve(row: pd.Series, points: pd.DataFrame, *, out_png: Path, ycol: str, g_type: str) -> None:
    pts = points.copy()
    pts = pts.sort_values(g_type, kind="stable")
    x = pd.to_numeric(pts[g_type], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(pts[ycol], errors="coerce").to_numpy(dtype=float)
    yhat = pd.to_numeric(pts["yhat"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if not mask.any():
        return
    x = x[mask]
    y = y[mask]
    yhat = yhat[mask]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(x, y, "o", color="#2c6fbb", label="data", markersize=5)
    if np.isfinite(yhat).any():
        order = np.argsort(x)
        ax.plot(x[order], yhat[order], "-", color="#c43c35", linewidth=2.0, label="fit")
    ax.set_xlabel(f"{g_type} corrected [mT/m]" if float(row.get("f_corr", 1.0) or 1.0) != 1.0 else f"{g_type} [mT/m]")
    ax.set_ylabel(str(ycol))
    ax.set_title(
        " | ".join(
            [
                f"{row.get('analysis_id')}",
                f"ROI={row.get('roi')}",
                f"dir={row.get('direction')}",
                f"N={float(row.get('N')):g}",
                f"td={float(row.get('td_ms')):g} ms",
            ]
        ),
        fontsize=10,
    )
    subtitle = (
        f"tc={float(row.get('tc_ms')):.4g} ms, alpha={float(row.get('alpha')):.4g}, "
        f"M0={float(row.get('M0')):.4g}, C={float(row.get('C')):.4g}, "
        f"f_corr={float(row.get('f_corr', 1.0) or 1.0):.5g}"
    )
    ax.text(0.01, 0.99, subtitle, transform=ax.transAxes, va="top", ha="left", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def write_per_analysis_outputs(
    *,
    fit_df: pd.DataFrame,
    signal_df: pd.DataFrame,
    points_df: pd.DataFrame,
    out_root: Path,
    model_name: str,
    ycol: str,
    g_type: str,
    write_csv: bool,
    write_signal_tables: bool,
) -> None:
    if fit_df.empty:
        return
    for analysis_id, sub in fit_df.groupby("analysis_id", sort=True):
        analysis_dir = out_root / str(analysis_id)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        fit_name = f"fit_params.{model_name}.{g_type}.{ycol}.parquet"
        fit_path = analysis_dir / fit_name
        write_table_outputs(
            sub.reset_index(drop=True),
            fit_path,
            xlsx_path=fit_path.with_suffix(".xlsx"),
            csv_path=fit_path.with_suffix(".csv") if write_csv else None,
        )

        sig_sub = signal_df[
            (signal_df["contrast_analysis_id"].astype(str) == str(analysis_id))
            | (signal_df["analysis_id"].astype(str) == str(analysis_id))
        ].copy()
        if write_signal_tables and not sig_sub.empty:
            sig_path = analysis_dir / f"signal_fit_params.{model_name}.{g_type}.{ycol}.parquet"
            write_table_outputs(
                sig_sub.reset_index(drop=True),
                sig_path,
                xlsx_path=sig_path.with_suffix(".xlsx"),
                csv_path=sig_path.with_suffix(".csv") if write_csv else None,
            )

        pts_sub = points_df[points_df["analysis_id"].astype(str) == str(analysis_id)].copy()
        if pts_sub.empty and not sig_sub.empty:
            sig_ids = set(sig_sub["analysis_id"].astype(str).tolist())
            pts_sub = points_df[points_df["analysis_id"].astype(str).isin(sig_ids)].copy()
        if write_signal_tables and not pts_sub.empty:
            pts_path = analysis_dir / f"signal_fit_points.{model_name}.{g_type}.{ycol}.parquet"
            write_table_outputs(
                pts_sub.reset_index(drop=True),
                pts_path,
                csv_path=pts_path.with_suffix(".csv") if write_csv else None,
            )

        for _, row in sig_sub.iterrows():
            mask = (
                (pts_sub["source_file"].astype(str) == str(row["source_file"]))
                & (pts_sub["roi"].astype(str) == str(row["roi"]))
                & (pts_sub["direction"].astype(str) == str(row["direction"]))
                & np.isclose(pd.to_numeric(pts_sub["td_ms"], errors="coerce"), float(row["td_ms"]), atol=1e-6)
                & np.isclose(pd.to_numeric(pts_sub["N"], errors="coerce"), float(row["N"]), atol=1e-6)
            )
            pts_curve = pts_sub[mask].copy()
            if pts_curve.empty:
                continue
            out_png = analysis_dir / (
                f"{sanitize_token(row['roi'])}.{sanitize_token(model_name)}."
                f"{sanitize_token(g_type)}.{sanitize_token(ycol)}."
                f"direction_{sanitize_token(row['direction'])}.png"
            )
            plot_curve(row, pts_curve, out_png=out_png, ycol=ycol, g_type=g_type)
