from __future__ import annotations

import repo_bootstrap  # noqa: F401

import argparse
from pathlib import Path
import re
import pandas as pd

from data_processing.io import write_table_outputs
from data_processing.master_table import append_master_rows, build_analysis_id_from_columns, load_master_table, select_signal
from fitting.b_from_g import VALID_G_TYPES as _GRADIENT_G_TYPES
from fitting.cli_common import signal_correction_factors_from_rows
from fitting.contrast import make_contrast
from tools.brain_labels import canonical_sheet_name, infer_subj_label
from tools.strict_columns import find_unrecognized_column_names
from tools.value_formatting import compact_unique_values, truthy_series

KEY_COLS = ("stat", "roi", "direction", "b_step")
SIGNAL_G_COLUMNS = ("g", "g_max", "g_lin_max", "g_thorsten", "bvalue", "bvalue_g", "bvalue_g_lin_max", "bvalue_thorsten")


def _normalize_direction_token(value: object) -> str:
    token = str(value).strip()
    if token == "":
        return ""
    try:
        num = float(token)
        if pd.notna(num) and abs(num - round(num)) < 1e-6:
            return str(int(round(num)))
    except Exception:
        pass
    return token


def _normalize_direction_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for raw in values:
        for token in str(raw).split(","):
            norm = _normalize_direction_token(token)
            if norm:
                out.append(norm)
    return list(dict.fromkeys(out))


def _split_values(values: list[str] | str | None) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return str(values).replace(",", " ").split() or None
    out: list[str] = []
    for value in values:
        out.extend(str(value).replace(",", " ").split())
    return out or None


def _one(df: pd.DataFrame, col: str, default=None):
    if col not in df.columns:
        return default
    u = pd.Series(df[col]).dropna().unique()
    return u[0] if len(u) else default


def _selector_value(value: object | None) -> object | None:
    if value is None:
        return None
    if isinstance(value, str):
        vals = _split_values(value)
        return vals if vals is not None and len(vals) > 1 else vals[0] if vals else None
    return value


def _master_shared_selectors(args: argparse.Namespace) -> dict[str, object]:
    selectors: dict[str, object] = {}
    for attr, col in [
        ("analysis_id", "analysis_id"),
        ("subj", "subj"),
        ("sheet", "sheet"),
        ("roi", "roi"),
        ("direction", "direction"),
        ("stat", "stat"),
        ("source_file", "source_file"),
    ]:
        value = _selector_value(getattr(args, attr, None))
        if value is not None:
            selectors[col] = value
    if args.td_ms is not None:
        selectors["td_ms"] = float(args.td_ms)
    return selectors


def _select_master_signal_side(
    master: pd.DataFrame,
    args: argparse.Namespace,
    *,
    side: int,
) -> pd.DataFrame:
    selectors = _master_shared_selectors(args)
    n_value = getattr(args, f"N_{side}")
    hz_value = getattr(args, f"Hz_{side}")
    g_value = getattr(args, f"g_{side}")
    if n_value is not None:
        selectors["N"] = float(n_value)
    if hz_value is not None:
        selectors["Hz"] = float(hz_value)
    if g_value is not None:
        g_col = str(args.g_pair_col)
        if g_col not in SIGNAL_G_COLUMNS:
            raise ValueError(f"--g_pair_col must be one of {sorted(SIGNAL_G_COLUMNS)}.")
        selectors[g_col] = float(g_value)

    df = select_signal(master, rotated=args.master_rotated, **selectors)
    if df.empty:
        raise ValueError(f"Master side {side} selection is empty. Selectors={selectors}")
    return df


def _master_analysis_id(df_ref: pd.DataFrame, df_cmp: pd.DataFrame, out: pd.DataFrame) -> str:
    preferred = ("subj", "sheet", "td_ms", "N_1", "N_2", "Hz_1", "Hz_2", "direction")
    try:
        return build_analysis_id_from_columns(out, columns=[c for c in preferred if c in out.columns], prefix="contrast")
    except ValueError:
        directions = sorted(out["direction"].dropna().astype(str).unique().tolist()) if "direction" in out.columns else []
        analysis_id, _ = build_analysis_id(df_ref, df_cmp, directions, None, oneg=False)
        return f"contrast_{analysis_id}"


def _fmt_num(x) -> str:
    if x is None:
        return "NA"
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not pd.notna(x):
        return "NA"
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _sanitize(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _fmt_seq(x) -> str:
    if x is None:
        return "NA"
    try:
        x = float(x)
    except Exception:
        return _sanitize(str(x))
    if not pd.notna(x):
        return "NA"
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return _sanitize(str(x))


def _has_oneg_marker(df: pd.DataFrame) -> bool:
    return "one_g_per_sequence" in df.columns and truthy_series(df["one_g_per_sequence"])


def _sequence_number(df: pd.DataFrame):
    seq = _one(df, "sequence", None)
    if seq is not None and str(seq).strip():
        return seq

    source = _one(df, "source_file", None)
    if source is None:
        return None

    m = re.search(r"_(\d+)_results(?:\.[A-Za-z0-9._-]+)?$", str(source))
    if m:
        return int(m.group(1))
    return None


def _sequence_label(df: pd.DataFrame, *, compact: bool = False) -> str:
    if compact and "sequence" in df.columns:
        values = pd.Series(df["sequence"]).dropna().unique().tolist()
        if values:
            return compact_unique_values(values)
    return _fmt_seq(_sequence_number(df))


def _build_analysis_core(
    df_ref: pd.DataFrame,
    df_cmp: pd.DataFrame,
    directions: list[str],
    sheet_override: str | None,
) -> tuple[str, str]:
    sheet = str(_one(df_ref, "sheet", _one(df_cmp, "sheet", "EXP")))
    if sheet_override:
        sheet = str(sheet_override)

    N1 = _one(df_ref, "N", None)
    N2 = _one(df_cmp, "N", None)
    try:
        N1i = int(round(float(N1))) if N1 is not None else -1
    except Exception:
        N1i = -1
    try:
        N2i = int(round(float(N2))) if N2 is not None else -1
    except Exception:
        N2i = -1

    td1 = _one(df_ref, "td_ms", None)
    hz1 = _one(df_ref, "Hz", None)
    hz2 = _one(df_cmp, "Hz", None)

    dir_tag = "-".join([str(d) for d in directions]) if directions else "ALL"
    td_tag = f"td{_fmt_num(td1)}" if (td1 is not None and pd.notna(td1)) else "tdNA"

    hz_tag = ""
    if hz1 is not None and pd.notna(hz1):
        hz_tag = f"_Hz{_fmt_num(hz1)}"
        if hz2 is not None and pd.notna(hz2) and abs(float(hz2) - float(hz1)) > 1e-6:
            hz_tag = f"_Hz{_fmt_num(hz1)}-{_fmt_num(hz2)}"

    analysis_core = f"{sheet}_N{N1i}-N{N2i}_{td_tag}{hz_tag}_dir{dir_tag}"
    analysis_short = f"{sheet}"
    return _sanitize(analysis_core)[:160], _sanitize(analysis_short)[:160]


def _validate_input(df: pd.DataFrame, label: str) -> None:
    unrecognized = find_unrecognized_column_names(df.columns)
    if unrecognized:
        raise ValueError(
            f"{label}: unrecognized column names: {unrecognized}. "
            "Use canonical names such as 'direction', 'value_norm', and 'g_thorsten'."
        )
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required key columns {missing}. Expected {KEY_COLS}.")


def _normalize_key_dtypes(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    for c in ["stat", "roi", "direction"]:
        out[c] = out[c].astype(str)
    out["direction"] = out["direction"].map(_normalize_direction_token)

    bs = pd.to_numeric(out["b_step"], errors="coerce")
    if bs.isna().any():
        bad = out.loc[bs.isna(), ["stat", "roi", "direction", "b_step"]].head(10)
        raise ValueError(f"{label}: b_step contains non-numeric values. Examples:\n{bad.to_string(index=False)}")
    out["b_step"] = bs.astype(int)
    return out


_NUMERIC_SUFFIX_RE = re.compile(r'_\d+(_\d+)*$')


def _merge_side_columns(out: pd.DataFrame, side_df: pd.DataFrame, *, side: int) -> pd.DataFrame:
    """
    Carry all columns from side_df except KEY_COLS, using the _1 or _2 suffix.
    Skip columns that already exist as {col}_{side}.
    Columns that already end in _N or _N_M (accumulated from prior contrast runs) are
    excluded to prevent exponential column growth on repeated contrast runs.
    """
    extra_cols = [
        c for c in side_df.columns
        if c not in KEY_COLS and not _NUMERIC_SUFFIX_RE.search(c)
    ]
    sub = side_df[list(KEY_COLS) + extra_cols].drop_duplicates(list(KEY_COLS), keep="first")

    rename = {}
    keep_extras = []
    for c in extra_cols:
        newc = f"{c}_{side}"
        if newc in out.columns:
            continue
        rename[c] = newc
        keep_extras.append(c)

    if not keep_extras:
        return out

    sub = sub[list(KEY_COLS) + keep_extras].rename(columns=rename)
    return out.merge(sub, on=list(KEY_COLS), how="left")


def _drop_aux_prefixed_cols(out: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in out.columns if c.startswith("param_") or c.startswith("meta_")]
    return out.drop(columns=drop_cols) if drop_cols else out


def _order_columns(out: pd.DataFrame) -> pd.DataFrame:
    """
    Final column order:
      roi, direction, b_step, stat,
      value, value_norm,
      [seq1: value_1, value_norm_1, S0_1, bvalues..., gradients..., params..., remaining],
      [seq2: ...],
      remaining unsuffixed columns
    """
    cols = list(out.columns)

    def present(xs):  # keep only existing columns while preserving order
        return [x for x in xs if x in cols]

    id_cols = present(["analysis_id", "subj", "sheet", "roi", "direction", "b_step", "stat"])
    head = id_cols + present(["value", "value_norm"])

    def side_block(suf: str) -> list[str]:
        block: list[str] = []
        # Core
        block += present([f"value{suf}", f"value_norm{suf}", f"S0{suf}"])

        # Put bvalue columns first
        b_pref = [
            f"bvalue{suf}",
            f"bvalue_g{suf}",
            f"bvalue_g_lin_max{suf}",
            f"bvalue_thorsten{suf}",
            f"bvalue_orig{suf}",
        ]
        block += present(b_pref)

        # Any remaining side-specific bvalue_* columns
        other_b = sorted([c for c in cols if c.endswith(suf) and c.startswith("bvalue") and c not in block])
        block += other_b

        # Gradients
        g_pref = [
            f"g{suf}",
            f"g_max{suf}",
            f"g_lin_max{suf}",
            f"g_thorsten{suf}",
        ]
        block += present(g_pref)

        other_g = sorted([c for c in cols if c.endswith(suf) and (c.startswith("g_") or c == f"g{suf}") and c not in block])
        block += other_g

        # Typical canonical parameters
        p_pref = [
            f"max_dur_ms{suf}", f"tm_ms{suf}", f"td_ms{suf}",
            f"Hz{suf}", f"N{suf}", f"TE{suf}", f"TR{suf}", f"bmax{suf}",
            f"protocol{suf}", f"sequence{suf}", f"sheet{suf}",
            f"Delta_app_ms{suf}", f"delta_ms{suf}",
            f"source_file{suf}",
        ]
        block += present(p_pref)

        # Remaining side-specific columns not yet included
        rest = sorted([c for c in cols if c.endswith(suf) and c not in block and c not in head])
        block += rest
        return block

    block1 = side_block("_1")
    block2 = side_block("_2")

    used = set(head + block1 + block2)
    tail = sorted([c for c in cols if c not in used])

    return out[head + block1 + block2 + tail]


def build_analysis_id(
    df_ref: pd.DataFrame,
    df_cmp: pd.DataFrame,
    directions: list[str],
    sheet_override: str | None,
    oneg: bool = False,
) -> tuple[str, str]:
    analysis_core, analysis_short = _build_analysis_core(df_ref, df_cmp, directions, sheet_override)
    compact_sequences = bool(oneg or _has_oneg_marker(df_ref) or _has_oneg_marker(df_cmp))
    seq1 = _sequence_label(df_ref, compact=compact_sequences)
    seq2 = _sequence_label(df_cmp, compact=compact_sequences)
    seq_tag = f"_seq{seq1}-{seq2}"
    analysis = f"{analysis_core}{seq_tag}"
    return _sanitize(analysis)[:160], analysis_short


def build_analysis_id_without_sequence(
    df_ref: pd.DataFrame,
    df_cmp: pd.DataFrame,
    directions: list[str],
    sheet_override: str | None,
) -> tuple[str, str]:
    return _build_analysis_core(df_ref, df_cmp, directions, sheet_override)


_GRADIENT_COLS = frozenset(_GRADIENT_G_TYPES)
_BVALUE_COLS = frozenset({"bvalue", "bvalue_g", "bvalue_g_lin_max", "bvalue_thorsten"})


def _apply_grad_correction(df: pd.DataFrame, f_by_direction: dict[str, float]) -> pd.DataFrame:
    """Scale gradient columns by f and bvalue columns by f² per direction."""
    out = df.copy()
    for direction, f in f_by_direction.items():
        if f == 1.0:
            continue
        mask = out["direction"].astype(str) == str(direction)
        for col in _GRADIENT_COLS:
            if col in out.columns:
                out.loc[mask, col] = pd.to_numeric(out.loc[mask, col], errors="coerce") * f
        f2 = f * f
        for col in _BVALUE_COLS:
            if col in out.columns:
                out.loc[mask, col] = pd.to_numeric(out.loc[mask, col], errors="coerce") * f2
    return out



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_parquet", nargs="?", help="signal parquet (ref)")
    ap.add_argument("cmp_parquet", nargs="?", help="signal parquet (cmp)")
    ap.add_argument("--direction", nargs="+", default=None, help="Filter by direction values, for example: 1 2 3 or long tra.")
    ap.add_argument("--subjs", nargs="+", default=None, help="Subjects/phantoms to include, for example: BRAIN-3 LUDG-2 PHANTOM3.")
    ap.add_argument("--out_root", default="analysis/ogse_experiments/contrast", help="directory root")
    ap.add_argument("--exp", default=None, help="Override the sheet name used for naming only.")
    ap.add_argument("--oneg", action="store_true", help="Allow one-g-per-sequence inputs and compact sequence labels.")
    ap.add_argument("--master-parquet", type=Path, default=None, help="Master table to select both contrast sides from.")
    ap.add_argument("--append-master", action="store_true", help="Append the contrast output back to --master-parquet with row_kind='contrast'.")
    ap.add_argument("--master-rotated", action=argparse.BooleanOptionalAction, default=True, help="Select rotated signal rows from master by default.")
    ap.add_argument("--analysis-id", action="append", default=None, help="Shared master analysis_id selector.")
    ap.add_argument("--subj", action="append", default=None, help="Shared master subj selector.")
    ap.add_argument("--sheet", action="append", default=None, help="Shared master sheet selector.")
    ap.add_argument("--roi", action="append", default=None, help="Shared master ROI selector.")
    ap.add_argument("--stat", action="append", default=None, help="Shared master stat selector.")
    ap.add_argument("--source-file", action="append", default=None, help="Shared master source_file selector.")
    ap.add_argument("--N_1", type=float, default=None, help="Master side-1 N selector.")
    ap.add_argument("--N_2", type=float, default=None, help="Master side-2 N selector.")
    ap.add_argument("--Hz_1", type=float, default=None, help="Master side-1 Hz selector.")
    ap.add_argument("--Hz_2", type=float, default=None, help="Master side-2 Hz selector.")
    ap.add_argument("--g_pair_col", default="g", choices=SIGNAL_G_COLUMNS, help="Column used by --g_1/--g_2 selectors.")
    ap.add_argument("--g_1", type=float, default=None, help="Optional master side-1 gradient selector using --g_pair_col.")
    ap.add_argument("--g_2", type=float, default=None, help="Optional master side-2 gradient selector using --g_pair_col.")
    ap.add_argument("--td_ms", type=float, default=None, help="Optional td_ms selector for master table rows.")
    corr_group = ap.add_mutually_exclusive_group()
    corr_group.add_argument(
        "--apply_grad_corr",
        action="store_true",
        help=(
            "Apply the gradient correction factor from master rows to each signal side before building the contrast. "
            "Gradient columns are rescaled per direction. "
            "Requires grad_correction_factor to be populated in master rows (run step 10 first)."
        ),
    )
    corr_group.add_argument("--no_grad_corr", action="store_true", help="Explicitly disable gradient correction (default).")
    args = ap.parse_args()

    if args.master_parquet is None and (args.ref_parquet is None or args.cmp_parquet is None):
        raise ValueError("Provide ref_parquet and cmp_parquet, or use --master-parquet with side selectors.")
    if args.append_master and args.master_parquet is None:
        raise ValueError("--append-master requires --master-parquet.")

    directions = _normalize_direction_list(args.direction)
    subjs = args.subjs
    if subjs is not None and len(subjs) == 1 and str(subjs[0]).upper() == "ALL":
        subjs = None

    if args.master_parquet is not None:
        master = load_master_table(args.master_parquet)
        df_ref = _select_master_signal_side(master, args, side=1)
        df_cmp = _select_master_signal_side(master, args, side=2)
    else:
        df_ref = pd.read_parquet(Path(args.ref_parquet))
        df_cmp = pd.read_parquet(Path(args.cmp_parquet))

    _validate_input(df_ref, "ref")
    _validate_input(df_cmp, "cmp")

    df_ref = _normalize_key_dtypes(df_ref, "ref")
    df_cmp = _normalize_key_dtypes(df_cmp, "cmp")

    if directions:
        ref_dirs_before = sorted(df_ref["direction"].astype(str).dropna().unique().tolist())
        cmp_dirs_before = sorted(df_cmp["direction"].astype(str).dropna().unique().tolist())
        df_ref = df_ref[df_ref["direction"].isin(directions)]
        df_cmp = df_cmp[df_cmp["direction"].isin(directions)]
        if df_ref.empty or df_cmp.empty:
            raise ValueError(
                "Direction filter left empty inputs. "
                f"Requested directions={directions}, ref_available={ref_dirs_before}, cmp_available={cmp_dirs_before}."
            )

    use_grad_corr = bool(args.apply_grad_corr) and not bool(args.no_grad_corr)
    if use_grad_corr:
        f_ref = signal_correction_factors_from_rows(df_ref)
        f_cmp = signal_correction_factors_from_rows(df_cmp)
        df_ref = _apply_grad_correction(df_ref, f_ref)
        df_cmp = _apply_grad_correction(df_cmp, f_cmp)

    oneg_mode = bool(args.oneg or _has_oneg_marker(df_ref) or _has_oneg_marker(df_cmp))

    analysis_id, analysis_short = build_analysis_id(df_ref, df_cmp, directions, args.exp, oneg=oneg_mode)
    old_analysis_id, _ = build_analysis_id_without_sequence(df_ref, df_cmp, directions, args.exp)
    sheet = canonical_sheet_name(args.exp or _one(df_ref, "sheet", _one(df_cmp, "sheet", None)))
    subj = _one(df_ref, "subj", _one(df_cmp, "subj", infer_subj_label(sheet, source_name=analysis_id)))

    if subjs is not None and str(subj) not in {str(x) for x in subjs}:
        print(f"Skipped: {analysis_id} (subj={subj})")
        return

    # Core contrast table: value/value_norm plus side-specific value_1/value_2 columns.
    res = make_contrast(
        df_ref,
        df_cmp,
        axes=tuple(directions) if directions else None,
        y_col="value",
        y_norm_col="value_norm",
        key_cols=KEY_COLS,
    )
    out = res.df.copy()

    _validate_input(out, "contrast_out")
    out = _normalize_key_dtypes(out, "contrast_out")

    # Carry all extra columns from ref and cmp.
    out = _merge_side_columns(out, df_ref, side=1)
    out = _merge_side_columns(out, df_cmp, side=2)

    _validate_input(out, "contrast_out")
    out = _normalize_key_dtypes(out, "contrast_out")

    # Strict cleanup
    out = _drop_aux_prefixed_cols(out)
    _validate_input(out, "contrast_clean")

    out["analysis_id"] = str(analysis_id)
    out["sheet"] = sheet
    out["subj"] = str(subj)

    # Final column order
    out = _order_columns(out)
    if args.master_parquet is not None and not out.empty:
        out["analysis_id"] = _master_analysis_id(df_ref, df_cmp, out)
        analysis_id = str(out["analysis_id"].iloc[0])
        out = _order_columns(out)

    tables_dir = Path(args.out_root) / "tables" / analysis_short
    tables_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = tables_dir / f"{analysis_id}.long.parquet"
    write_table_outputs(out, out_parquet, xlsx_path=out_parquet.with_suffix(".xlsx"))
    if args.append_master:
        append_master_rows(
            args.master_parquet,
            out,
            row_kind="contrast",
            analysis_id=str(analysis_id),
            out_path=args.master_parquet,
        )
        print("Appended contrast to master:", args.master_parquet)

    # Remove older duplicate outputs that used the pre-sequence naming scheme.
    if old_analysis_id != analysis_id:
        old_parquet = tables_dir / f"{old_analysis_id}.long.parquet"
        old_xlsx = old_parquet.with_suffix(".xlsx")
        for old_path in (old_parquet, old_xlsx):
            if old_path.exists():
                old_path.unlink()
                print("Removed duplicate output:", old_path)

    print("Saved:", out_parquet)


if __name__ == "__main__":
    main()
