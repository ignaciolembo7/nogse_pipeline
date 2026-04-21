from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd

from ogse_fitting.contrast import make_contrast
from tools.brain_labels import canonical_sheet_name, infer_subj_label
from tools.strict_columns import find_unrecognized_column_names

KEY_COLS = ("stat", "roi", "direction", "b_step")


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


def _one(df: pd.DataFrame, col: str, default=None):
    if col not in df.columns:
        return default
    u = pd.Series(df[col]).dropna().unique()
    return u[0] if len(u) else default


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


def _compact_values(values: list[object]) -> str:
    text_values = [str(value) for value in values]
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    if numeric.notna().all():
        nums = sorted({float(value) for value in numeric.to_numpy(dtype=float)})
        if all(float(value).is_integer() for value in nums):
            ints = [int(value) for value in nums]
            if ints and ints == list(range(ints[0], ints[-1] + 1)):
                return f"{ints[0]}-{ints[-1]}"
            return ",".join(str(value) for value in ints)
        return ",".join(f"{value:g}" for value in nums)
    return ",".join(text_values)


def _truthy_series(series: pd.Series) -> bool:
    values = series.dropna().astype(str).str.strip().str.lower()
    return values.isin(["1", "true", "yes", "y", "on"]).any()


def _has_oneg_marker(df: pd.DataFrame) -> bool:
    return "one_g_per_sequence" in df.columns and _truthy_series(df["one_g_per_sequence"])


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
            return _compact_values(values)
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


def _merge_side_columns(out: pd.DataFrame, side_df: pd.DataFrame, *, side: int) -> pd.DataFrame:
    """
    Carry all columns from side_df except KEY_COLS, using the _1 or _2 suffix.
    Skip columns that already exist as {col}_{side}.
    """
    extra_cols = [c for c in side_df.columns if c not in KEY_COLS]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_parquet", help="signal parquet (ref)")
    ap.add_argument("cmp_parquet", help="signal parquet (cmp)")
    ap.add_argument("--direction", nargs="+", default=None, help="Filter by direction values, for example: 1 2 3 or long tra.")
    ap.add_argument("--subjs", nargs="+", default=None, help="Subjects/phantoms to include, for example: BRAIN-3 LUDG-2 PHANTOM3.")
    ap.add_argument("--out_root", default="analysis/ogse_experiments/contrast", help="directory root")
    ap.add_argument("--exp", default=None, help="Override the sheet name used for naming only.")
    ap.add_argument("--oneg", action="store_true", help="Allow one-g-per-sequence inputs and compact sequence labels.")
    args = ap.parse_args()

    directions = _normalize_direction_list(args.direction)
    subjs = args.subjs
    if subjs is not None and len(subjs) == 1 and str(subjs[0]).upper() == "ALL":
        subjs = None

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

    # Carry all extra columns from ref and cmp
    out = _merge_side_columns(out, df_ref, side=1)
    out = _merge_side_columns(out, df_cmp, side=2)

    # Strict cleanup
    out = _drop_aux_prefixed_cols(out)
    _validate_input(out, "contrast_clean")

    out["analysis_id"] = str(analysis_id)
    out["sheet"] = sheet
    out["subj"] = str(subj)

    # Final column order
    out = _order_columns(out)

    tables_dir = Path(args.out_root) / "tables" / analysis_short
    tables_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = tables_dir / f"{analysis_id}.long.parquet"
    out.to_parquet(out_parquet, index=False)
    out.to_excel(out_parquet.with_suffix(".xlsx"), index=False)

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
