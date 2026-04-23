from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data_processing.io import write_table_outputs
from data_processing.schema import finalize_clean_dproj_long
from fitting.b_from_g import b_from_g
from fitting.core import CurveFitParameter
from fitting.core import chi2 as _chi2
from fitting.core import fit_curve_fit_parameters
from fitting.core import rmse as _rmse
from fitting.core import rmse_log as _rmse_log
from ogse_plotting.plot_ogse_signal_vs_g import plot_ogse_signal_fit
from tools.fit_params_schema import standardize_fit_params
from tools.strict_columns import raise_on_unrecognized_column_names


AUTO_FIT_MIN_POINTS = 3
AUTO_FIT_MAX_POINTS = 9
AUTO_FIT_REL_TOL = 0.05
AUTO_FIT_ERR_FLOOR = 5e-3
AUTO_FIT_ABS_TOL = 1e-6

GTYPE_ALIASES = {
    "bvalue": "bvalue",
    "g": "g",
    "bvalue_g": "g",
    "g_max": "g_max",
    "g_lin_max": "g_lin_max",
    "bvalue_g_lin_max": "g_lin_max",
    "g_thorsten": "g_thorsten",
    "bvalue_thorsten": "g_thorsten",
}
VALID_G_TYPES = set(GTYPE_ALIASES)


def monoexp(b: np.ndarray, M0: float, D0: float) -> np.ndarray:
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
                raise ValueError(f"Expected one unique value in {c!r} for {name}, found: {v[:10]}")
            return float(v[0])
    if required:
        raise ValueError(f"Could not infer {name}. Checked columns: {list(cols)}")
    return None


def _unique_str(df: pd.DataFrame, col: str) -> Optional[str]:
    if col not in df.columns:
        return None
    u = pd.Series(df[col]).dropna().astype(str).unique()
    if len(u) == 1:
        return str(u[0])
    return None


def _require_supported_b_axis_for_fit(df: pd.DataFrame, *, g_type: str) -> None:
    gradient_axis_kind = _unique_str(df, "gradient_axis_kind")
    if gradient_axis_kind is None or gradient_axis_kind.lower() != "g":
        return

    requested = _normalize_g_type(g_type)
    if requested == "bvalue":
        target_cols = ["bvalue"]
    elif requested == "g":
        target_cols = ["bvalue_g"]
    elif requested == "g_lin_max":
        target_cols = ["bvalue_g_lin_max"]
    elif requested == "g_thorsten":
        target_cols = ["bvalue_thorsten"]
    else:
        target_cols = ["bvalue"]

    has_supported_axis = any(
        col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any()
        for col in target_cols
    )
    if has_supported_axis:
        return

    raise ValueError(
        "This fit step still requires a real b-value axis. "
        "The input was marked as direct g-only data, but the needed "
        f"{target_cols} column(s) are empty."
    )


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


def build_monoexp_dproj_long(
    df_signal: pd.DataFrame,
    fit_params: pd.DataFrame,
    *,
    stat_keep: str = "avg",
) -> pd.DataFrame:
    signal = df_signal.copy()
    if "stat" in signal.columns and stat_keep is not None and str(stat_keep).upper() != "ALL":
        signal = signal[signal["stat"].astype(str) == str(stat_keep)].copy()

    empty_cols = [
        "roi", "direction", "b_step", "bvalue", "D_proj", "source_file",
        "subj", "max_dur_ms", "tm_ms", "td_ms", "Hz", "N", "TE", "TR",
        "bmax", "protocol", "sequence", "sheet", "Delta_app_ms", "delta_ms",
    ]
    if signal.empty:
        return pd.DataFrame(columns=empty_cols)

    subj_by_key: dict[tuple[str, str], str] = {}
    if not fit_params.empty and {"roi", "direction"}.issubset(fit_params.columns):
        for _, fit_row in fit_params.iterrows():
            fit_subj = str(fit_row.get("subj", "")).strip()
            if fit_subj and fit_subj.lower() not in {"nan", "none", "<na>"}:
                subj_by_key[(str(fit_row["roi"]), str(fit_row["direction"]))] = fit_subj

    rows: list[dict] = []
    for _, signal_row in signal.iterrows():
        bvalue = pd.to_numeric(pd.Series([signal_row.get("bvalue")]), errors="coerce").iloc[0]
        if not np.isfinite(bvalue) or bvalue <= 0:
            continue

        value_norm = pd.to_numeric(pd.Series([signal_row.get("value_norm")]), errors="coerce").iloc[0]
        if not np.isfinite(value_norm):
            value = pd.to_numeric(pd.Series([signal_row.get("value")]), errors="coerce").iloc[0]
            s0 = pd.to_numeric(pd.Series([signal_row.get("S0")]), errors="coerce").iloc[0]
            if np.isfinite(value) and np.isfinite(s0) and s0 != 0.0:
                value_norm = float(value / s0)
        if not np.isfinite(value_norm):
            continue

        value_norm = float(np.clip(value_norm, 1e-12, None))
        dproj = float(-np.log(value_norm) / float(bvalue))

        row = signal_row.to_dict()
        row["D_proj"] = dproj

        key = (str(row.get("roi")), str(row.get("direction")))
        fit_subj = subj_by_key.get(key, "").strip()
        if fit_subj:
            row["subj"] = fit_subj

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=empty_cols)

    return finalize_clean_dproj_long(pd.DataFrame(rows))


def _require_strict_columns(df: pd.DataFrame) -> None:
    raise_on_unrecognized_column_names(df.columns, context="fit_ogse_signal_vs_g")


def _ensure_keys_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["direction"] = out["direction"].astype(str)
    out["roi"] = out["roi"].astype(str)
    bs = pd.to_numeric(out["b_step"], errors="coerce")
    if bs.isna().any():
        bad = out.loc[bs.isna(), ["roi", "direction", "b_step"]].head(10)
        raise ValueError(f"b_step contains non-numeric values. Examples:\n{bad.to_string(index=False)}")
    out["b_step"] = bs.astype(int)
    if "stat" in out.columns:
        out["stat"] = out["stat"].astype(str)
    return out


def _normalize_g_type(g_type: str) -> str:
    raw = str(g_type).strip()
    if raw not in VALID_G_TYPES:
        raise ValueError(
            f"fit_ogse_signal_vs_g: unrecognized g_type {raw!r}. "
            f"Allowed values: {sorted(VALID_G_TYPES)}."
        )
    return GTYPE_ALIASES.get(raw, raw)


def _b_from_mode(
    d: pd.DataFrame,
    *,
    g_type: str,
    gamma: float,
    N: Optional[float],
    delta_ms: Optional[float],
    Delta_app_ms: Optional[float],
) -> np.ndarray:
    g_type = _normalize_g_type(g_type)

    if g_type == "bvalue":
        if "bvalue" not in d.columns:
            raise ValueError("Missing required column 'bvalue'.")
        return pd.to_numeric(d["bvalue"], errors="coerce").to_numpy(dtype=float)

    bcol_map = {"g": "bvalue_g", "g_lin_max": "bvalue_g_lin_max", "g_thorsten": "bvalue_thorsten"}
    bcol = bcol_map.get(g_type)
    if bcol and bcol in d.columns:
        return pd.to_numeric(d[bcol], errors="coerce").to_numpy(dtype=float)

    if g_type not in d.columns:
        raise ValueError(f"Missing required column {g_type!r}; derived column {bcol!r} was also not found.")

    if N is None or delta_ms is None or Delta_app_ms is None:
        raise ValueError("For g_type != 'bvalue', N, delta_ms, and Delta_app_ms must be provided as arguments or unique columns.")

    g = pd.to_numeric(d[g_type], errors="coerce").to_numpy(dtype=float)
    return b_from_g(
        g,
        N=float(N),
        gamma=float(gamma),
        delta_ms=float(delta_ms),
        delta_app_ms=float(Delta_app_ms),
        g_type=g_type,
    )


@dataclass(frozen=True)
class FitOutputs:
    fit_params: pd.DataFrame
    fit_table: pd.DataFrame


def _fit_prefix_monoexp(
    b: np.ndarray,
    y: np.ndarray,
    *,
    fit_points: int,
    free_M0: bool,
    fix_M0: float,
    D0_init: float,
) -> dict:
    k = min(int(fit_points), len(b))
    if k <= 0:
        return {
            "ok": False,
            "fit_points": int(k),
            "n_fit": 0,
            "fit_mask": np.zeros(len(b), dtype=bool),
            "msg": "fit_points must be > 0.",
        }

    prefix_mask = np.zeros(len(b), dtype=bool)
    prefix_mask[:k] = True
    valid_mask = np.isfinite(b) & np.isfinite(y) & (y > 0)
    fit_mask = prefix_mask & valid_mask

    b_fit = b[fit_mask].copy()
    y_fit = y[fit_mask].copy()

    result = {
        "ok": False,
        "fit_points": int(k),
        "n_fit": int(len(b_fit)),
        "fit_mask": fit_mask,
        "msg": "",
    }

    if len(b_fit) < AUTO_FIT_MIN_POINTS:
        result["msg"] = "Too few valid points."
        return result

    try:
        method_label = "curve_fit(M0,D0)" if free_M0 else "curve_fit(D0) M0_fixed"

        def model_from_params(params: dict[str, float]) -> np.ndarray:
            return monoexp(b_fit, float(params["M0"]), float(params["D0_mm2_s"]))

        fit = fit_curve_fit_parameters(
            model_from_params,
            y_fit,
            parameters=[
                CurveFitParameter("M0", 1.0 if free_M0 else float(fix_M0), 0.0, 100.0, bool(free_M0)),
                CurveFitParameter("D0_mm2_s", float(D0_init), D0_init / 10, 2 * D0_init, True),
            ],
            x=b_fit,
            maxfev=40000,
            method=method_label,
        )
        M0_hat = float(fit.values["M0"])
        D0_hat = float(fit.values["D0_mm2_s"])
        M0_err = float(fit.errors.get("M0_err", np.nan)) if free_M0 else np.nan
        D0_err = float(fit.errors.get("D0_err_mm2_s", np.nan))
        method = fit.method
    except Exception as exc:
        result["msg"] = f"Falló curve_fit: {exc}"
        return result

    yhat = monoexp(b_fit, M0_hat, D0_hat)
    result.update(
        ok=True,
        msg="",
        M0=float(M0_hat),
        M0_err=float(M0_err) if np.isfinite(M0_err) else np.nan,
        D0_mm2_s=float(D0_hat),
        D0_err_mm2_s=float(D0_err) if np.isfinite(D0_err) else np.nan,
        D0_m2_ms=float(D0_hat) * 1e-9,
        D0_err_m2_ms=float(D0_err) * 1e-9 if np.isfinite(D0_err) else np.nan,
        rmse=float(_rmse(y_fit, yhat)),
        chi2=float(_chi2(y_fit, yhat)),
        rmse_log=float(_rmse_log(y_fit, yhat)),
        method=method,
    )
    return result


def _select_fit_result(
    b: np.ndarray,
    y: np.ndarray,
    *,
    fit_points: Optional[int],
    auto_fit_points: bool,
    free_M0: bool,
    fix_M0: float,
    D0_init: float,
    auto_fit_min_points: int = AUTO_FIT_MIN_POINTS,
    auto_fit_max_points: Optional[int] = AUTO_FIT_MAX_POINTS,
    auto_fit_rel_tol: float = AUTO_FIT_REL_TOL,
    auto_fit_err_floor: float = AUTO_FIT_ERR_FLOOR,
) -> dict:
    if auto_fit_points:
        k_min = max(1, int(auto_fit_min_points))
        k_max = len(b) if auto_fit_max_points is None else min(int(auto_fit_max_points), len(b))
        if k_max < k_min:
            return {
                "ok": False,
                "fit_points": np.nan,
                "n_fit": 0,
                "fit_strategy": "auto",
                "auto_fit_metric": "rmse_log",
                "auto_fit_score": np.nan,
                "msg": f"Invalid auto_fit_points range: min_k={k_min}, max_k={k_max}.",
                "fit_mask": np.zeros(len(b), dtype=bool),
            }

        last_ok = None
        stop_msg = None
        tested_until = k_min - 1
        for k in range(k_min, k_max + 1):
            tested_until = k
            cand = _fit_prefix_monoexp(
                b,
                y,
                fit_points=k,
                free_M0=free_M0,
                fix_M0=fix_M0,
                D0_init=D0_init,
            )
            if not cand["ok"]:
                if last_ok is None:
                    continue
                stop_msg = f"Stopped at k={k} because the fit became invalid: {cand.get('msg', 'invalid fit')}"
                break

            if last_ok is None:
                last_ok = cand
                continue

            prev_score = float(last_ok["rmse_log"])
            curr_score = float(cand["rmse_log"])
            effective_prev_score = max(prev_score, float(auto_fit_err_floor))
            allowed_score = effective_prev_score * (1.0 + auto_fit_rel_tol) + AUTO_FIT_ABS_TOL
            if curr_score <= allowed_score:
                last_ok = cand
                continue

            stop_msg = (
                f"Stopped at k={k}: rmse_log={curr_score:.4g} exceeded "
                f"allowed={allowed_score:.4g} from previous k={int(last_ok['fit_points'])} "
                f"(prev={prev_score:.4g}, floor={float(auto_fit_err_floor):.4g}, tol={auto_fit_rel_tol:.2%})."
            )
            break

        if last_ok is not None:
            selected = dict(last_ok)
            selected["fit_strategy"] = "auto"
            selected["auto_fit_metric"] = "rmse_log"
            selected["auto_fit_score"] = float(selected["rmse_log"])
            if stop_msg is None:
                stop_msg = f"Reached max k={tested_until} within tolerance (tol={auto_fit_rel_tol:.2%})."
            selected["msg"] = (
                f"Auto fit_points selected {int(selected['fit_points'])} "
                f"after testing k={k_min}..{tested_until}. {stop_msg}"
            )
            selected["method"] = (
                f"{selected['method']} | "
                f"auto_fit_points_sequential(rmse_log, tol={auto_fit_rel_tol:.2%}, "
                f"err_floor={float(auto_fit_err_floor):.4g}, min_k={k_min}, max_k={k_max})"
            )
            return selected

        valid_total = int(np.sum(np.isfinite(b) & np.isfinite(y) & (y > 0)))
        return {
            "ok": False,
            "fit_points": np.nan,
            "n_fit": valid_total,
            "fit_strategy": "auto",
            "auto_fit_metric": "rmse_log",
            "auto_fit_score": np.nan,
            "msg": f"No valid candidate was found for auto_fit_points in the range k={k_min}..{k_max}.",
            "fit_mask": np.zeros(len(b), dtype=bool),
        }

    selected_fit_points = fit_points if fit_points is not None else 6
    selected = _fit_prefix_monoexp(
        b,
        y,
        fit_points=int(selected_fit_points),
        free_M0=free_M0,
        fix_M0=fix_M0,
        D0_init=D0_init,
    )
    selected["fit_strategy"] = "fixed"
    selected["auto_fit_metric"] = np.nan
    selected["auto_fit_score"] = np.nan
    return selected


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
        raise KeyError(f"plot_fit_one_group_monoexp: missing ycol {ycol!r}. Columns={list(df_group.columns)}")

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
    f_corr = float(fit_row.get("f_corr", 1.0) or 1.0)
    b_corr_scale = float(f_corr) ** 2.0
    if np.isfinite(b_corr_scale):
        b = b * b_corr_scale

    mask = np.isfinite(y) & np.isfinite(b)
    y = y[mask]
    b = b[mask]
    if len(b) == 0:
        raise ValueError("No valid points are available for plotting.")

    bmax = float(np.nanmax(b)) if np.any(np.isfinite(b)) else 0.0
    b_dense = np.linspace(0.0, bmax, 250)

    ys = None
    if bool(fit_row.get("ok", True)):
        M0 = float(fit_row["M0"])
        D0 = float(fit_row["D0_mm2_s"])
        ys = monoexp(b_dense, M0, D0)
    plot_ogse_signal_fit(
        b=np.asarray(b, dtype=float),
        y=np.asarray(y, dtype=float),
        fit_row=fit_row,
        out_png=out_png,
        ycol=ycol,
        fit_points=int(fit_points),
        fit_x=np.asarray(b_dense, dtype=float) if ys is not None else None,
        fit_y=np.asarray(ys, dtype=float) if ys is not None else None,
    )


def fit_ogse_signal_vs_g_long(
    df: pd.DataFrame,
    *,
    dirs: Optional[Sequence[str]] = None,
    roi: str = "ALL",
    rois: Optional[Sequence[str]] = None,
    ycol: str = "value_norm",
    fit_points: Optional[int] = 6,
    auto_fit_points: bool = False,
    auto_fit_min_points: int = AUTO_FIT_MIN_POINTS,
    auto_fit_max_points: Optional[int] = AUTO_FIT_MAX_POINTS,
    auto_fit_rel_tol: float = AUTO_FIT_REL_TOL,
    auto_fit_err_floor: float = AUTO_FIT_ERR_FLOOR,
    g_type: str = "bvalue",
    gamma: float = 267.5221900,
    N: Optional[float] = None,
    delta_ms: Optional[float] = None,
    Delta_app_ms: Optional[float] = None,
    td_ms: float = np.nan,
    D0_init: float = 0.0023,
    free_M0: bool = False,
    fix_M0: float = 1.0,
    f_by_direction: Mapping[str, float] | None = None,
    b_corr_power: float = 2.0,
    outdir_plots: Optional[Path] = None,
    title_prefix: str = "",
    stat_keep: str = "avg",
) -> FitOutputs:
    del title_prefix

    dfa = df.copy()
    _require_strict_columns(dfa)

    for c in ["direction", "roi", "b_step"]:
        if c not in dfa.columns:
            raise ValueError(f"Missing required column {c!r}. Columns={list(dfa.columns)}")

    if ycol not in {"value", "value_norm"}:
        raise ValueError("ycol must be one of ['value', 'value_norm'].")
    if ycol not in dfa.columns:
        raise ValueError(f"Missing ycol {ycol!r}. Columns={list(dfa.columns)}")

    dfa = _ensure_keys_types(dfa)

    if "stat" in dfa.columns and stat_keep is not None and str(stat_keep).upper() != "ALL":
        dfa = dfa[dfa["stat"] == str(stat_keep)].copy()
        if dfa.empty:
            raise ValueError(f"No rows remain after filtering stat == {stat_keep!r}.")

    if dirs is not None:
        dfa = dfa[dfa["direction"].isin([str(x) for x in dirs])].copy()
    rois_keep = _normalize_requested_rois(roi=roi, rois=rois)
    if rois_keep is not None:
        dfa = dfa[dfa["roi"].isin(rois_keep)].copy()
    if dfa.empty:
        dirs_avail = sorted(df["direction"].astype(str).dropna().unique().tolist()) if "direction" in df.columns else []
        rois_avail = sorted(df["roi"].astype(str).dropna().unique().tolist()) if "roi" in df.columns else []
        raise ValueError(
            "No rows remain after filtering by direction/roi. "
            f"Available directions={dirs_avail}. Available ROIs={rois_avail}."
        )

    if outdir_plots is not None:
        outdir_plots.mkdir(parents=True, exist_ok=True)

    results = []
    points = []

    for (dir_val, roi_val), d in dfa.groupby(["direction", "roi"], sort=False):
        d = d.sort_values("b_step", kind="stable").copy()

        y = pd.to_numeric(d[ycol], errors="coerce").to_numpy(dtype=float)
        b = _b_from_mode(d, g_type=g_type, gamma=gamma, N=N, delta_ms=delta_ms, Delta_app_ms=Delta_app_ms)
        f_corr = float(f_by_direction.get(str(dir_val), 1.0)) if f_by_direction else 1.0
        b_corr_scale = float(f_corr) ** float(b_corr_power)
        if np.isfinite(b_corr_scale):
            b = b * b_corr_scale

        fit_res = _select_fit_result(
            b,
            y,
            fit_points=fit_points,
            auto_fit_points=auto_fit_points,
            free_M0=free_M0,
            fix_M0=fix_M0,
            D0_init=D0_init,
            auto_fit_min_points=auto_fit_min_points,
            auto_fit_max_points=auto_fit_max_points,
            auto_fit_rel_tol=auto_fit_rel_tol,
            auto_fit_err_floor=auto_fit_err_floor,
        )

        selected_fit_points = fit_res.get("fit_points")
        selected_fit_points_out = (
            int(selected_fit_points) if selected_fit_points is not None and np.isfinite(selected_fit_points) else np.nan
        )

        if not fit_res["ok"]:
            results.append(
                dict(
                    roi=str(roi_val),
                    direction=str(dir_val),
                    stat=str(stat_keep),
                    model="monoexp",
                    ycol=str(ycol),
                    g_type=str(g_type),
                    fit_points=selected_fit_points_out,
                    fit_strategy=str(fit_res.get("fit_strategy", "fixed")),
                    auto_fit_metric=fit_res.get("auto_fit_metric", np.nan),
                    auto_fit_score=fit_res.get("auto_fit_score", np.nan),
                    rmse_log=fit_res.get("rmse_log", np.nan),
                    f_corr=float(f_corr),
                    b_corr_scale=float(b_corr_scale),
                    n_points=int(len(b)),
                    n_fit=int(fit_res.get("n_fit", 0)),
                    td_ms=float(td_ms),
                    ok=False,
                    msg=str(fit_res.get("msg", "Too few valid points.")),
                )
            )
            continue

        results.append(
            dict(
                roi=str(roi_val),
                direction=str(dir_val),
                stat=str(stat_keep),
                model="monoexp",
                ycol=str(ycol),
                g_type=str(g_type),
                fit_points=selected_fit_points_out,
                fit_strategy=str(fit_res.get("fit_strategy", "fixed")),
                auto_fit_metric=fit_res.get("auto_fit_metric", np.nan),
                auto_fit_score=fit_res.get("auto_fit_score", np.nan),
                f_corr=float(f_corr),
                b_corr_scale=float(b_corr_scale),
                n_points=int(len(b)),
                n_fit=int(fit_res["n_fit"]),
                td_ms=float(td_ms),
                N=float(N) if N is not None else np.nan,
                delta_ms=float(delta_ms) if delta_ms is not None else np.nan,
                Delta_app_ms=float(Delta_app_ms) if Delta_app_ms is not None else np.nan,
                M0=float(fit_res["M0"]),
                M0_err=fit_res["M0_err"],
                D0_mm2_s=float(fit_res["D0_mm2_s"]),
                D0_err_mm2_s=fit_res["D0_err_mm2_s"],
                D0_m2_ms=float(fit_res["D0_m2_ms"]),
                D0_err_m2_ms=fit_res["D0_err_m2_ms"],
                rmse=float(fit_res["rmse"]),
                rmse_log=float(fit_res["rmse_log"]),
                chi2=float(fit_res["chi2"]),
                method=str(fit_res["method"]),
                ok=True,
                msg=str(fit_res.get("msg", "")),
            )
        )

        used = fit_res["fit_mask"]
        for bs, bi, yi, uu in zip(d["b_step"].to_numpy(), b, y, used):
            points.append(
                dict(
                    roi=str(roi_val),
                    direction=str(dir_val),
                    stat=str(stat_keep),
                    ycol=str(ycol),
                    g_type=str(g_type),
                    fit_points=selected_fit_points_out,
                    td_ms=float(td_ms),
                    f_corr=float(f_corr),
                    b_corr_scale=float(b_corr_scale),
                    b_step=int(bs),
                    bvalue_used=float(bi) if np.isfinite(bi) else np.nan,
                    y=float(yi) if np.isfinite(yi) else np.nan,
                    used_for_fit=bool(uu),
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
                fit_points=int(selected_fit_points_out) if np.isfinite(selected_fit_points_out) else 0,
            )

    return FitOutputs(fit_params=pd.DataFrame(results), fit_table=pd.DataFrame(points))


def run_fit_ogse_signal_vs_g_from_parquet(
    parquet_path: str | Path,
    *,
    dirs: Optional[Sequence[str]] = None,
    roi: str = "ALL",
    rois: Optional[Sequence[str]] = None,
    ycol: str = "value_norm",
    g_type: str = "bvalue",
    fit_points: Optional[int] = 6,
    auto_fit_points: bool = False,
    auto_fit_min_points: int = AUTO_FIT_MIN_POINTS,
    auto_fit_max_points: Optional[int] = AUTO_FIT_MAX_POINTS,
    auto_fit_rel_tol: float = AUTO_FIT_REL_TOL,
    auto_fit_err_floor: float = AUTO_FIT_ERR_FLOOR,
    free_M0: bool = False,
    fix_M0: float = 1.0,
    D0_init: float = 0.0023,
    gamma: float = 267.5221900,
    N: Optional[float] = None,
    delta_ms: Optional[float] = None,
    Delta_app_ms: Optional[float] = None,
    td_ms: Optional[float] = None,
    stat_keep: str = "avg",
    out_root: str | Path = "ogse_experiments/fits/ogse_signal_vs_g_monoexp",
    out_dproj_root: Optional[str | Path] = None,
    f_by_direction: Mapping[str, float] | None = None,
    b_corr_power: float = 2.0,
) -> Tuple[FitOutputs, Path]:
    p = Path(parquet_path)
    df = pd.read_parquet(p) if p.suffix.lower() not in [".xlsx", ".xls"] else pd.read_excel(p, sheet_name=0)

    _require_strict_columns(df)
    df = _ensure_keys_types(df)
    _require_supported_b_axis_for_fit(df, g_type=g_type)

    exp_id = infer_exp_id(p)
    sheet = _unique_str(df, "sheet") or exp_id.split("_")[0]

    exp_dir = Path(out_root) / sheet / exp_id
    tables_dir = exp_dir
    plots_dir = exp_dir

    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    outs = fit_ogse_signal_vs_g_long(
        df,
        dirs=dirs,
        roi=roi,
        rois=rois,
        ycol=ycol,
        fit_points=fit_points,
        auto_fit_points=auto_fit_points,
        auto_fit_min_points=auto_fit_min_points,
        auto_fit_max_points=auto_fit_max_points,
        auto_fit_rel_tol=auto_fit_rel_tol,
        auto_fit_err_floor=auto_fit_err_floor,
        g_type=g_type,
        gamma=gamma,
        N=N,
        delta_ms=delta_ms,
        Delta_app_ms=Delta_app_ms,
        td_ms=td_ms,
        D0_init=D0_init,
        free_M0=free_M0,
        fix_M0=fix_M0,
        f_by_direction=f_by_direction,
        b_corr_power=b_corr_power,
        outdir_plots=plots_dir,
        title_prefix=f"{exp_id} | ",
        stat_keep=stat_keep,
    )

    fp = outs.fit_params.copy()
    if not fp.empty:
        fp["subj"] = _unique_str(df, "subj") if "subj" in df.columns else np.nan
        fp["max_dur_ms"] = _unique_float_any(df, ["max_dur_ms"], required=False, name="max_dur_ms")
        fp["tm_ms"] = _unique_float_any(df, ["tm_ms"], required=False, name="tm_ms")
        fp["td_ms"] = td_ms if td_ms is not None else _unique_float_any(df, ["td_ms"], required=False, name="td_ms")
        fp = standardize_fit_params(fp, fit_kind="monoexp", source_file=p.name)

    out_params_parquet = tables_dir / "fit_params.parquet"
    out_params_xlsx = tables_dir / "fit_params.xlsx"
    out_points_parquet = tables_dir / "fit_points.parquet"
    out_points_xlsx = tables_dir / "fit_points.xlsx"

    write_table_outputs(fp, out_params_parquet, xlsx_path=out_params_xlsx)
    write_table_outputs(outs.fit_table, out_points_parquet, xlsx_path=out_points_xlsx)

    if out_dproj_root is not None and not fp.empty:
        dproj_dir = Path(out_dproj_root) / sheet
        dproj_dir.mkdir(parents=True, exist_ok=True)
        dproj = build_monoexp_dproj_long(df, fp, stat_keep=stat_keep)
        if not dproj.empty:
            out_dproj_parquet = dproj_dir / f"{exp_id}.monoexp.Dproj.long.parquet"
            out_dproj_xlsx = dproj_dir / f"{exp_id}.monoexp.Dproj.long.xlsx"
            write_table_outputs(dproj, out_dproj_parquet, xlsx_path=out_dproj_xlsx)

    return outs, exp_dir


def run_fit_from_parquet(
    parquet_path: str | Path,
    *,
    dirs: Optional[Sequence[str]] = None,
    roi: str = "ALL",
    rois: Optional[Sequence[str]] = None,
    ycol: str = "value_norm",
    g_type: str = "bvalue",
    fit_points: Optional[int] = 6,
    auto_fit_points: bool = False,
    auto_fit_min_points: int = AUTO_FIT_MIN_POINTS,
    auto_fit_max_points: Optional[int] = AUTO_FIT_MAX_POINTS,
    auto_fit_rel_tol: float = AUTO_FIT_REL_TOL,
    auto_fit_err_floor: float = AUTO_FIT_ERR_FLOOR,
    free_M0: bool = False,
    fix_M0: float = 1.0,
    D0_init: float = 0.0023,
    gamma: float = 267.5221900,
    N: Optional[float] = None,
    delta_ms: Optional[float] = None,
    Delta_app_ms: Optional[float] = None,
    td_ms: Optional[float] = None,
    stat_keep: str = "avg",
    out_root: str | Path = "ogse_experiments/fits/ogse_signal_vs_g_monoexp",
    out_dproj_root: Optional[str | Path] = None,
    f_by_direction: Mapping[str, float] | None = None,
    b_corr_power: float = 2.0,
) -> Tuple[FitOutputs, Path]:
    return run_fit_ogse_signal_vs_g_from_parquet(
        parquet_path,
        dirs=dirs,
        roi=roi,
        rois=rois,
        ycol=ycol,
        g_type=g_type,
        fit_points=fit_points,
        auto_fit_points=auto_fit_points,
        auto_fit_min_points=auto_fit_min_points,
        auto_fit_max_points=auto_fit_max_points,
        auto_fit_rel_tol=auto_fit_rel_tol,
        auto_fit_err_floor=auto_fit_err_floor,
        free_M0=free_M0,
        fix_M0=fix_M0,
        D0_init=D0_init,
        gamma=gamma,
        N=N,
        delta_ms=delta_ms,
        Delta_app_ms=Delta_app_ms,
        td_ms=td_ms,
        stat_keep=stat_keep,
        out_root=out_root,
        out_dproj_root=out_dproj_root,
        f_by_direction=f_by_direction,
        b_corr_power=b_corr_power,
    )


__all__ = [
    "AUTO_FIT_ABS_TOL",
    "AUTO_FIT_ERR_FLOOR",
    "AUTO_FIT_MAX_POINTS",
    "AUTO_FIT_MIN_POINTS",
    "AUTO_FIT_REL_TOL",
    "FitOutputs",
    "VALID_G_TYPES",
    "build_monoexp_dproj_long",
    "fit_ogse_signal_vs_g_long",
    "infer_exp_id",
    "monoexp",
    "run_fit_from_parquet",
    "run_fit_ogse_signal_vs_g_from_parquet",
]
