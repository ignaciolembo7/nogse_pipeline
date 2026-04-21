from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from data_processing.io import infer_layout_from_filename, read_result_xls
from data_processing.match_params import parse_results_filename, select_params_row
from data_processing.params import read_sequence_params_xlsx
from data_processing.reshape import to_long
from data_processing.schema import finalize_clean_signal_long
from ogse_fitting.b_from_g import b_from_g


# -------------------------
# Helpers
# -------------------------
def _norm_key(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _row_get(row: pd.Series, keys: list[str], default=None):
    """Busca en row por keys (case/space-insensitive) y devuelve el primer valor no-NaN."""
    idx = list(row.index)
    norm_map = {_norm_key(c): c for c in idx}

    for k in keys:
        kk = _norm_key(k)
        if kk in norm_map:
            col = norm_map[kk]
            v = row[col]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return v
    return default


def _to_float(v):
    if v is None:
        return np.nan
    try:
        return float(v)
    except Exception:
        return np.nan


def _ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _first_present(columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def _sanitize_token(value) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "group"


def _split_strip_tokens(values: list[str] | None) -> list[str]:
    tokens: list[str] = []
    for value in values or []:
        tokens.extend(t for t in re.split(r"[,\s]+", str(value)) if t)
    return tokens


def _output_strip_tokens(cli_tokens: list[str] | None) -> list[str]:
    env_tokens = os.environ.get("NOGSE_OUTPUT_STEM_STRIP", "")
    return _split_strip_tokens([env_tokens]) + _split_strip_tokens(cli_tokens)


def _strip_output_tokens(stem: str, tokens: list[str]) -> str:
    out = str(stem)
    for token in tokens:
        out = out.replace(token, "")
    return re.sub(r"_+", "_", out).strip("_")


def _output_stem_from_results_file(
    results_file: Path,
    *,
    grouped_g_curve: bool,
    strip_tokens: list[str],
) -> str:
    stem = results_file.stem
    if grouped_g_curve:
        stem = re.sub(r"_G\d+(?:p\d+|\.\d+)?(?=_)", "", stem, flags=re.IGNORECASE)
        stem = re.sub(r"_\d+(?=_results$)", "", stem, flags=re.IGNORECASE)
    stem = _strip_output_tokens(stem, strip_tokens)
    return _sanitize_token(stem)


def _truthy_env(name: str) -> bool:
    raw = os.environ.get(name, "")
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _finite_float(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _normalize_protocol_label(protocol, group_value, tn_value) -> str | None:
    if protocol is None or not str(protocol).strip():
        return None

    base = str(protocol).strip()
    base = re.sub(r"(^|_)BW\d+(?=_|$)", r"\1", base, flags=re.IGNORECASE)
    base = re.sub(r"(^|_)G\d+(?:p\d+|\.\d+)?(?=_|$)", r"\1", base, flags=re.IGNORECASE)

    group_float = _finite_float(group_value)
    if group_float is not None:
        group_num = int(round(group_float))
        updated, nsubs = re.subn(r"Exp\d+", f"Exp{group_num:02d}", base, count=1, flags=re.IGNORECASE)
        base = updated if nsubs else f"{base}_group{group_num:03d}"

    tn_float = _finite_float(tn_value)
    if tn_float is not None:
        base = re.sub(r"TN\d+(?:p\d+|\.\d+)?", f"TN{int(round(tn_float))}", base, count=1, flags=re.IGNORECASE)

    base = re.sub(r"_+", "_", base).strip("_")
    return base or None


@dataclass(frozen=True)
class CleanSequenceParams:
    sheet: str
    subj: str
    protocol: str | None
    sequence: object
    group: object
    G: float
    TN: float
    x: float
    y: float
    type: object
    Hz: float
    bmax: float
    N: int
    delta_ms: float
    Delta_app_ms: float
    max_dur_ms: float
    tm_ms: float
    td_ms: float
    TE: float
    TR: float
    g_thorsten: float | None

    def as_columns(self) -> dict[str, object]:
        out = {
            "subj": self.subj,
            "sheet": self.sheet,
            "protocol": self.protocol,
            "sequence": self.sequence,
            "group": self.group,
            "G": self.G,
            "TN": self.TN,
            "x": self.x,
            "y": self.y,
            "type": self.type,
            "Hz": self.Hz,
            "bmax": self.bmax,
            "N": self.N,
            "delta_ms": self.delta_ms,
            "Delta_app_ms": self.Delta_app_ms,
            "max_dur_ms": self.max_dur_ms,
            "tm_ms": self.tm_ms,
            "td_ms": self.td_ms,
            "TE": self.TE,
            "TR": self.TR,
        }
        if self.g_thorsten is not None:
            out["g_thorsten"] = self.g_thorsten
        return out


def _detect_gradient_input_kind(stats: dict[str, pd.DataFrame]) -> str:
    any_df = next(iter(stats.values()))
    cols = [str(c) for c in any_df.columns]

    if _first_present(cols, ["gvalues", "gval", "g"]) is not None:
        return "g"
    if _first_present(cols, ["bvalues", "bvalue", "bval", "b"]) is not None:
        return "b"
    raise ValueError(f"Could not detect a supported gradient column in results table. Columns={cols}")


def _is_single_point_results(stats: dict[str, pd.DataFrame]) -> bool:
    return all(len(df.index) == 1 for df in stats.values())


def _infer_project_root(path: Path) -> Path:
    resolved = path.resolve()
    for parent in [resolved.parent] + list(resolved.parents):
        if (parent / "Data-NIFTI").is_dir():
            return parent
        if parent.name == "Data-signals":
            return parent.parent
    return resolved.parent


def _find_matching_gradient_vector(results_file: Path) -> Path | None:
    seq_name = results_file.name.removesuffix("_results.xlsx")
    project_root = _infer_project_root(results_file)
    search_root = project_root / "Data-NIFTI"
    if not search_root.is_dir():
        return None

    for ext in (".gvec", ".bvec"):
        matches = sorted(search_root.rglob(f"{seq_name}{ext}"))
        if matches:
            return matches[0]
    return None


def _direction_from_vector_path(vector_path: Path | None) -> str:
    if vector_path is None:
        return "1"

    arr = np.loadtxt(vector_path, dtype=float)
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        if arr.size != 3:
            raise ValueError(f"Expected 3 values or 3xN values in {vector_path}, got shape={arr.shape}")
        vec = arr
    else:
        if arr.shape[0] != 3:
            raise ValueError(f"Expected 3 rows in {vector_path}, got shape={arr.shape}")
        vec = np.nanmean(arr, axis=1)

    avec = np.abs(vec)
    axis = int(np.nanargmax(avec))
    if not np.isfinite(avec[axis]):
        return "1"

    return str(axis + 1)


def _aggregate_g_results(
    stats: dict[str, pd.DataFrame],
    *,
    direction_label: str,
    source_file: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    g_value_ref: float | None = None

    for stat, df in stats.items():
        gcol = _first_present(list(df.columns), ["gvalues", "gval", "g"])
        if gcol is None:
            raise ValueError(f"Missing g-values column in sheet/stat {stat}. Columns={list(df.columns)}")

        rois = [c for c in df.columns if c != gcol]
        gvals = pd.to_numeric(df[gcol], errors="coerce").dropna().unique().tolist()
        if len(gvals) != 1:
            raise ValueError(
                f"Expected a single unique g value in sheet/stat {stat}, found {gvals}."
            )

        g_value = float(gvals[0])
        if g_value_ref is None:
            g_value_ref = g_value
        elif not np.isclose(g_value_ref, g_value, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"Inconsistent g values across stat sheets: first={g_value_ref}, current={g_value} ({stat})."
            )

        summary = df[rois].apply(pd.to_numeric, errors="coerce").mean(axis=0)
        b_step = 0 if np.isclose(g_value, 0.0, rtol=0.0, atol=1e-9) else 1

        for roi in rois:
            rows.append(
                {
                    "stat": stat,
                    "roi": str(roi),
                    "direction": str(direction_label),
                    "b_step": int(b_step),
                    "bvalue": np.nan,
                    "g": float(g_value),
                    "value": float(summary.loc[roi]),
                    "source_file": source_file,
                    "gradient_axis_kind": "g",
                }
            )

    return pd.DataFrame(rows)


def _assign_group_bsteps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g_numeric = pd.to_numeric(out["g"], errors="coerce")
    unique_g = np.sort(pd.unique(g_numeric.dropna()))

    step_map: dict[float, int] = {}
    next_step = 1
    for gv in unique_g:
        if np.isclose(float(gv), 0.0, rtol=0.0, atol=1e-9):
            step_map[float(gv)] = 0
        else:
            step_map[float(gv)] = next_step
            next_step += 1

    def _step_for_value(v) -> int:
        fv = pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0]
        if not np.isfinite(fv):
            return pd.NA
        for key, step in step_map.items():
            if np.isclose(float(fv), float(key), rtol=0.0, atol=1e-9):
                return int(step)
        return pd.NA

    out["b_step"] = out["g"].map(_step_for_value)
    return out


def _add_direct_g_derivatives(df: pd.DataFrame, *, gamma: float) -> pd.DataFrame:
    out = df.copy()
    out = _ensure_numeric(out, ["g", "b_step", "N", "delta_ms", "Delta_app_ms", "g_thorsten"])

    if "g" not in out.columns:
        return out

    g_numeric = pd.to_numeric(out["g"], errors="coerce")
    if g_numeric.dropna().empty:
        return out

    gmax = out.groupby("direction")["g"].transform("max")
    out["g_max"] = gmax

    b_step = pd.to_numeric(out["b_step"], errors="coerce")
    b_step_max = b_step.groupby(out["direction"]).transform("max")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["g_lin_max"] = gmax * (b_step / b_step_max)
    out.loc[b_step_max <= 0, "g_lin_max"] = g_numeric

    if "g_thorsten" not in out.columns:
        out["g_thorsten"] = pd.NA

    n_value = _single_numeric_value(out, "N")
    delta_ms = _single_numeric_value(out, "delta_ms")
    delta_app_ms = _single_numeric_value(out, "Delta_app_ms")
    if n_value is None or delta_ms is None or delta_app_ms is None:
        for col in ["bvalue_g", "bvalue_g_lin_max", "bvalue_thorsten"]:
            if col not in out.columns:
                out[col] = pd.NA
        return out

    out["bvalue_g"] = b_from_g(
        pd.to_numeric(out["g"], errors="coerce").to_numpy(float),
        N=n_value,
        gamma=gamma,
        delta_ms=delta_ms,
        delta_app_ms=delta_app_ms,
        g_type="g",
    )
    out["bvalue_g_lin_max"] = b_from_g(
        pd.to_numeric(out["g_lin_max"], errors="coerce").to_numpy(float),
        N=n_value,
        gamma=gamma,
        delta_ms=delta_ms,
        delta_app_ms=delta_app_ms,
        g_type="g_lin_max",
    )
    out["bvalue_thorsten"] = b_from_g(
        pd.to_numeric(out["g_thorsten"], errors="coerce").to_numpy(float),
        N=n_value,
        gamma=gamma,
        delta_ms=delta_ms,
        delta_app_ms=delta_app_ms,
        g_type="g_thorsten",
    )
    return out


def _single_numeric_value(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    values = pd.to_numeric(df[col], errors="coerce").dropna().unique()
    if len(values) != 1:
        return None
    return float(values[0])


def _single_text_value(df: pd.DataFrame, col: str) -> str | None:
    if col not in df.columns:
        return None
    values = df[col].dropna().astype(str).str.strip().unique()
    values = [v for v in values if v]
    if len(values) != 1:
        return None
    return str(values[0])


def _filter_existing_to_incoming_curve(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    out = existing.copy()

    for col in ["sheet", "subj", "type"]:
        value = _single_text_value(incoming, col)
        if value is not None and col in out.columns:
            out = out.loc[out[col].astype(str).str.strip() == value].copy()

    for col in ["group", "TN", "N"]:
        value = _single_numeric_value(incoming, col)
        if value is not None and col in out.columns:
            existing_values = pd.to_numeric(out[col], errors="coerce")
            out = out.loc[np.isclose(existing_values.to_numpy(float), value, rtol=0.0, atol=1e-6)].copy()

    return out


def _merge_into_group_curve(existing: pd.DataFrame | None, incoming: pd.DataFrame, *, gamma: float) -> pd.DataFrame:
    if existing is None or existing.empty:
        merged = incoming.copy()
    else:
        existing = _filter_existing_to_incoming_curve(existing, incoming)
        src_files = set(incoming["source_file"].astype(str).tolist())
        keep_existing = ~existing["source_file"].astype(str).isin(src_files)
        merged = pd.concat([existing.loc[keep_existing].copy(), incoming.copy()], ignore_index=True)

    merged = _assign_group_bsteps(merged)
    merged = _add_direct_g_derivatives(merged, gamma=gamma)
    drop_cols = [c for c in ["S0", "value_norm"] if c in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
    merged = _add_S0_and_value_norm(merged)
    merged = finalize_clean_signal_long(merged)
    merged = merged.sort_values(["stat", "roi", "direction", "b_step", "g"], kind="stable").reset_index(drop=True)
    return merged


def _extract_clean_sequence_params(row: pd.Series, meta) -> CleanSequenceParams:
    sheet = str(_row_get(row, ["sheet"], meta.sheet))
    subj = _row_get(row, ["subj"], None)
    if subj is None or not str(subj).strip():
        raise ValueError(
            "Matched sequence_parameters row is missing a required 'subj' value. "
            f"sheet={sheet!r}, seq={meta.seq}"
        )
    subj = str(subj).strip()

    group_value = _row_get(row, ["group"], getattr(meta, "group", None))
    g_value = _to_float(_row_get(row, ["G"], getattr(meta, "G", np.nan)))
    tn_value = _to_float(_row_get(row, ["TN"], getattr(meta, "TN", np.nan)))
    protocol_raw = _row_get(row, ["protocol", "Protocol*"], None)
    protocol = _normalize_protocol_label(protocol_raw, group_value, tn_value)

    max_dur_ms = _to_float(_row_get(row, ["max_dur_ms", "d_ms", "Max duration d  [ms]"], None))
    tm_ms = _to_float(_row_get(row, ["tm_ms", "TM_ms", "mixing time TM  [ms]"], None))
    td_ms = _to_float(_row_get(row, ["td_ms", "TN"], np.nan))
    if not np.isfinite(td_ms) and np.isfinite(max_dur_ms) and np.isfinite(tm_ms):
        td_ms = 2.0 * max_dur_ms + tm_ms

    g_thorsten = _to_float(_row_get(row, ["g_thorsten", "G thorsten [mT/m]"], np.nan))
    if not np.isfinite(g_thorsten):
        g_thorsten = None

    return CleanSequenceParams(
        sheet=sheet,
        subj=subj,
        protocol=protocol,
        sequence=_row_get(row, ["sequence", "seq"], None),
        group=group_value,
        G=g_value,
        TN=tn_value,
        x=_to_float(_row_get(row, ["x"], np.nan)),
        y=_to_float(_row_get(row, ["y"], np.nan)),
        type=_row_get(row, ["type", "seq_type"], None),
        Hz=_to_float(_row_get(row, ["Hz", "Frecuency [Hz]"], meta.Hz)),
        bmax=_to_float(_row_get(row, ["bmax", "bval_max [s/mm2]"], meta.bmax)),
        N=int(_to_float(_row_get(row, ["N"], 1))),
        delta_ms=_to_float(_row_get(row, ["delta_ms", "delta [ms]"], meta.delta_ms)),
        Delta_app_ms=_to_float(_row_get(row, ["Delta_app_ms", "delta_app_ms", "Delta_app [ms]"], meta.Delta_ms)),
        max_dur_ms=max_dur_ms,
        tm_ms=tm_ms,
        td_ms=td_ms,
        TE=_to_float(_row_get(row, ["TE", "TE_ms", "Echo time TE  [ms]"], np.nan)),
        TR=_to_float(_row_get(row, ["TR", "TR_ms", "Repetition time TR  [ms]"], np.nan)),
        g_thorsten=g_thorsten,
    )


def _add_clean_sequence_params(df: pd.DataFrame, params: CleanSequenceParams) -> pd.DataFrame:
    out = df.copy()
    for col, value in params.as_columns().items():
        out[col] = value
    return out


def _add_S0_and_value_norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = _ensure_numeric(out, ["b_step", "value"])

    # S0 per (stat, roi, direction): mean value where b_step == 0.
    s0 = (
        out.loc[out["b_step"] == 0]
        .groupby(["stat", "roi", "direction"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "S0"})
    )
    out = out.merge(s0, on=["stat", "roi", "direction"], how="left")

    with np.errstate(divide="ignore", invalid="ignore"):
        out["value_norm"] = out["value"] / out["S0"]

    # Force 1.0 at b_step == 0 when S0 is available.
    m0 = (out["b_step"] == 0) & out["S0"].notna()
    out.loc[m0, "value_norm"] = 1.0
    return out


def _add_g_and_b_derivatives(
    df: pd.DataFrame,
    *,
    gamma: float,
    N: int,
    delta_ms: float,
    Delta_app_ms: float,
    g_thorsten: float | None,
) -> pd.DataFrame:
    out = df.copy()
    out = _ensure_numeric(out, ["bvalue", "b_step"])

    # Compute g from bvalue (mT/m).
    b = out["bvalue"].to_numpy(dtype=float)  # s/mm^2
    denom = (N * (gamma**2) * (delta_ms**2) * (Delta_app_ms))
    g = np.sqrt((b * 1e9) / denom)
    g[np.isclose(b, 0.0)] = 0.0
    out["g"] = g

    # g_max by direction, using the last b_step.
    b_step_max = int(np.nanmax(out["b_step"].to_numpy(dtype=float)))
    if not np.isfinite(b_step_max) or b_step_max <= 0:
        b_step_max = 1

    gmax_by_dir = (
        out.loc[out["b_step"] == b_step_max]
        .groupby("direction")["g"]
        .max()
    )
    out["g_max"] = out["direction"].map(gmax_by_dir)
    # Fallback for any direction without a match.
    out["g_max"] = out["g_max"].fillna(np.nanmax(out["g"].to_numpy(dtype=float)))

    # g_lin_max
    out["g_lin_max"] = out["g_max"] * (out["b_step"] / float(b_step_max))

    # g_thorsten, when the scalar is available.
    if g_thorsten is None or not np.isfinite(g_thorsten):
        out["g_thorsten"] = np.nan
    else:
        out["g_thorsten"] = float(g_thorsten) * (out["b_step"] / float(b_step_max))

    # bvalue columns derived from g* columns.
    out["bvalue_g"] = b_from_g(
        pd.to_numeric(out["g"], errors="coerce").to_numpy(float),
        N=N,
        gamma=gamma,
        delta_ms=delta_ms,
        delta_app_ms=Delta_app_ms,
        g_type="g",
    )
    out["bvalue_g_lin_max"] = b_from_g(
        pd.to_numeric(out["g_lin_max"], errors="coerce").to_numpy(float),
        N=N,
        gamma=gamma,
        delta_ms=delta_ms,
        delta_app_ms=Delta_app_ms,
        g_type="g_lin_max",
    )
    # Thorsten can be NaN; b_from_g propagates NaN.
    out["bvalue_thorsten"] = b_from_g(
        pd.to_numeric(out["g_thorsten"], errors="coerce").to_numpy(float),
        N=N,
        gamma=gamma,
        delta_ms=delta_ms,
        delta_app_ms=Delta_app_ms,
        g_type="g_thorsten",
    )

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_file", type=Path)
    ap.add_argument("params_xlsx", type=Path)
    ap.add_argument("--out_dir", type=Path, default=Path("analysis/ogse_experiments/data"))
    ap.add_argument("--gamma", type=float, default=267.5221900, help="1/(ms*mT)")
    ap.add_argument(
        "--oneg",
        action="store_true",
        default=_truthy_env("NOGSE_ONEG"),
        help="Treat one-g-per-sequence result files as points of the same direct-g curve.",
    )
    ap.add_argument(
        "--strip-output-token",
        "--output-stem-strip",
        dest="strip_output_tokens",
        action="append",
        default=[],
        help="Token to remove from generated output stems. Can be repeated or comma/space separated.",
    )
    args = ap.parse_args()

    stats = read_result_xls(args.results_file)
    gradient_input_kind = _detect_gradient_input_kind(stats)
    oneg_mode = bool(args.oneg)

    meta = parse_results_filename(args.results_file)
    if gradient_input_kind == "b":
        layout = infer_layout_from_filename(args.results_file)
        ndirs = meta.ndirs or layout.ndirs
        nbvals = meta.nbvals or layout.nbvals
        if ndirs is None or nbvals is None:
            raise SystemExit(f"No pude inferir ndirs/nbvals del filename: {args.results_file.name}")
        df_long = to_long(stats, ndirs=ndirs, nbvals=nbvals, source_file=args.results_file.name)
    else:
        if _is_single_point_results(stats):
            vector_path = _find_matching_gradient_vector(args.results_file)
            direction_label = _direction_from_vector_path(vector_path)
            df_long = _aggregate_g_results(
                stats,
                direction_label=direction_label,
                source_file=args.results_file.name,
            )
        else:
            layout = infer_layout_from_filename(args.results_file)
            ndirs = meta.ndirs or layout.ndirs
            nbvals = meta.nbvals or layout.nbvals
            if ndirs is None or nbvals is None:
                raise SystemExit(f"No pude inferir ndirs/nbvals del filename: {args.results_file.name}")

            any_df = next(iter(stats.values()))
            gcol = _first_present([str(c) for c in any_df.columns], ["gvalues", "gval", "g"])
            if gcol is None:
                raise SystemExit(f"No pude encontrar una columna g en: {list(any_df.columns)}")

            df_long = to_long(
                stats,
                ndirs=ndirs,
                nbvals=nbvals,
                bcol=gcol,
                out_col="g",
                source_file=args.results_file.name,
            )
            df_long["bvalue"] = np.nan

    # --- params (single matched row) ---
    params = read_sequence_params_xlsx(args.params_xlsx)
    row = select_params_row(params, meta)
    clean_params = _extract_clean_sequence_params(row, meta)

    df_long = _add_clean_sequence_params(df_long, clean_params)
    df_long["gradient_axis_kind"] = gradient_input_kind

    # --- features + normalization ---
    if gradient_input_kind == "b":
        df_long = _add_g_and_b_derivatives(
            df_long,
            gamma=float(args.gamma),
            N=int(clean_params.N),
            delta_ms=float(clean_params.delta_ms),
            Delta_app_ms=float(clean_params.Delta_app_ms),
            g_thorsten=clean_params.g_thorsten,
        )
        df_long = _add_S0_and_value_norm(df_long)
    else:
        if "g_thorsten" not in df_long.columns:
            df_long["g_thorsten"] = pd.NA
        df_long["bvalue_g"] = pd.NA
        df_long["bvalue_g_lin_max"] = pd.NA
        df_long["bvalue_thorsten"] = pd.NA
        df_long = _add_direct_g_derivatives(df_long, gamma=float(args.gamma))
        df_long["S0"] = pd.NA
        df_long["value_norm"] = pd.NA
        zero_mask = np.isclose(pd.to_numeric(df_long["g"], errors="coerce"), 0.0, rtol=0.0, atol=1e-9)
        df_long.loc[zero_mask, "S0"] = pd.to_numeric(df_long.loc[zero_mask, "value"], errors="coerce")
        df_long.loc[zero_mask, "value_norm"] = 1.0

    df_long["one_g_per_sequence"] = oneg_mode
    df_long = finalize_clean_signal_long(df_long)
    df_long = df_long.sort_values(["stat", "roi", "direction", "b_step"], kind="stable").reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = args.out_dir / clean_params.sheet
    exp_dir.mkdir(parents=True, exist_ok=True)

    is_grouped_g_curve = oneg_mode and gradient_input_kind == "g" and _is_single_point_results(stats)
    if oneg_mode and not is_grouped_g_curve:
        raise ValueError("--oneg expects a direct-g results file with one row per stat sheet.")
    out_stem = _output_stem_from_results_file(
        args.results_file,
        grouped_g_curve=is_grouped_g_curve,
        strip_tokens=_output_strip_tokens(args.strip_output_tokens),
    )
    out_path = exp_dir / f"{out_stem}.long.parquet"

    if is_grouped_g_curve:
        existing = pd.read_parquet(out_path) if out_path.exists() else None
        df_to_save = _merge_into_group_curve(existing, df_long, gamma=float(args.gamma))
    else:
        df_to_save = df_long

    df_to_save.to_parquet(out_path, index=False)
    df_to_save.to_excel(out_path.with_suffix(".xlsx"), index=False)

    print("Selected params (clean):")
    print(
        pd.Series(
            {
                "sheet": clean_params.sheet,
                "subj": clean_params.subj,
                "protocol": clean_params.protocol,
                "sequence": clean_params.sequence,
                "group": clean_params.group,
                "G": clean_params.G,
                "TN": clean_params.TN,
                "x": clean_params.x,
                "y": clean_params.y,
                "type": clean_params.type,
                "Hz": clean_params.Hz,
                "bmax": clean_params.bmax,
                "N": clean_params.N,
                "delta_ms": clean_params.delta_ms,
                "Delta_app_ms": clean_params.Delta_app_ms,
                "max_dur_ms": clean_params.max_dur_ms,
                "tm_ms": clean_params.tm_ms,
                "td_ms": clean_params.td_ms,
                "TE": clean_params.TE,
                "TR": clean_params.TR,
            }
        ).to_string()
    )
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
