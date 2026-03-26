from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd

from ogse_fitting.contrast import make_contrast
from tools.brain_labels import canonical_sheet_name, infer_brain_group

KEY_COLS = ("stat", "roi", "direction", "b_step")


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


def _validate_input(df: pd.DataFrame, label: str) -> None:
    if "axis" in df.columns:
        raise ValueError(
            f"[{label}] Encontré 'axis'. Este pipeline usa SOLO 'direction'. Arreglá upstream."
        )
    missing = [c for c in KEY_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{label}] Faltan columnas clave {missing}. Necesito {KEY_COLS}.")


def _normalize_key_dtypes(df: pd.DataFrame, label: str) -> pd.DataFrame:
    out = df.copy()
    for c in ["stat", "roi", "direction"]:
        out[c] = out[c].astype(str)

    bs = pd.to_numeric(out["b_step"], errors="coerce")
    if bs.isna().any():
        bad = out.loc[bs.isna(), ["stat", "roi", "direction", "b_step"]].head(10)
        raise ValueError(f"[{label}] b_step inválido. Ejemplos:\n{bad.to_string(index=False)}")
    out["b_step"] = bs.astype(int)
    return out


def _merge_side_columns(out: pd.DataFrame, side_df: pd.DataFrame, *, side: int) -> pd.DataFrame:
    """
    Arrastra TODAS las columnas de side_df (excepto KEY_COLS) con sufijo _1 o _2.
    Evita duplicar si ya existe {col}_{side}.
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


def _drop_legacy_cols(out: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [c for c in out.columns if c.startswith("param_") or c.startswith("meta_")]
    return out.drop(columns=drop_cols) if drop_cols else out


def _dedup_aliases(out: pd.DataFrame) -> pd.DataFrame:
    """
    Si aparecen alias por transición histórica, nos quedamos con el canónico.
    (Esto evita "parámetros repetidos" con distinto nombre.)
    """
    o = out.copy()

    # Delta_app_ms canónico (si aparece delta_app_ms también)
    for suf in ("_1", "_2"):
        low = f"delta_app_ms{suf}"
        can = f"Delta_app_ms{suf}"
        if can in o.columns and low in o.columns:
            o = o.drop(columns=[low])

    # g_thorsten canónico (si aparece gthorsten también)
    for suf in ("_1", "_2"):
        bad = f"gthorsten{suf}"
        can = f"g_thorsten{suf}"
        if can in o.columns and bad in o.columns:
            o = o.drop(columns=[bad])

        badb = f"bvalue_gthorsten{suf}"
        canb = f"bvalue_thorsten{suf}"
        if canb in o.columns and badb in o.columns:
            o = o.drop(columns=[badb])

    return o


def _rename_to_generic(out: pd.DataFrame) -> pd.DataFrame:
    """
    - en el resultado del contraste: ya vienen 'value' y 'value_norm' desde la lib
    - para las secuencias: renombramos signal_norm_{1,2} -> value_norm_{1,2}
      (para que NO aparezca 'signal' en la tabla final)
    """
    o = out.copy()
    ren = {}
    if "signal_norm_1" in o.columns and "value_norm_1" not in o.columns:
        ren["signal_norm_1"] = "value_norm_1"
    if "signal_norm_2" in o.columns and "value_norm_2" not in o.columns:
        ren["signal_norm_2"] = "value_norm_2"
    if ren:
        o = o.rename(columns=ren)
    return o


def _order_columns(out: pd.DataFrame) -> pd.DataFrame:
    """
    Orden final pedido:
      roi, direction, b_step, stat,
      value, value_norm,
      [seq1: value_1, value_norm_1, S0_1, bvalues..., gradients..., params..., resto],
      [seq2: ...],
      resto (no sufijado)
    """
    cols = list(out.columns)

    def present(xs):  # filtra por existentes manteniendo orden
        return [x for x in xs if x in cols]

    id_cols = present(["analysis_id", "brain", "sheet", "roi", "direction", "b_step", "stat"])
    head = id_cols + present(["value", "value_norm"])

    def side_block(suf: str) -> list[str]:
        block: list[str] = []
        # core
        block += present([f"value{suf}", f"value_norm{suf}", f"S0{suf}"])

        # bvalues primero
        b_pref = [
            f"bvalue{suf}",
            f"bvalue_g{suf}",
            f"bvalue_g_lin_max{suf}",
            f"bvalue_thorsten{suf}",
            f"bvalue_orig{suf}",
        ]
        block += present(b_pref)

        # cualquier otro bvalue_* del lado
        other_b = sorted([c for c in cols if c.endswith(suf) and c.startswith("bvalue") and c not in block])
        block += other_b

        # gradients
        g_pref = [
            f"g{suf}",
            f"g_max{suf}",
            f"g_lin_max{suf}",
            f"g_thorsten{suf}",
        ]
        block += present(g_pref)

        other_g = sorted([c for c in cols if c.endswith(suf) and (c.startswith("g_") or c == f"g{suf}") and c not in block])
        block += other_g

        # parámetros típicos (canónicos)
        p_pref = [
            f"max_dur_ms{suf}", f"tm_ms{suf}", f"td_ms{suf}",
            f"Hz{suf}", f"N{suf}", f"TE{suf}", f"TR{suf}", f"bmax{suf}",
            f"protocol{suf}", f"sequence{suf}", f"sheet{suf}",
            f"Delta_app_ms{suf}", f"delta_ms{suf}",
            f"source_file{suf}",
        ]
        block += present(p_pref)

        # resto del lado (cualquier cosa *_1 / *_2 no incluida)
        rest = sorted([c for c in cols if c.endswith(suf) and c not in block and c not in head])
        block += rest
        return block

    block1 = side_block("_1")
    block2 = side_block("_2")

    used = set(head + block1 + block2)
    tail = sorted([c for c in cols if c not in used])

    return out[head + block1 + block2 + tail]


def build_analysis_id(df_ref: pd.DataFrame, df_cmp: pd.DataFrame, directions: list[str], sheet_override: str | None) -> str:
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

    analysis = f"{sheet}_N{N1i}-N{N2i}_{td_tag}{hz_tag}_dir{dir_tag}"
    analysis_short = f"{sheet}"
    return _sanitize(analysis)[:160], _sanitize(analysis_short)[:160]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref_parquet", help="signal parquet (ref)")
    ap.add_argument("cmp_parquet", help="signal parquet (cmp)")
    ap.add_argument("--direction", nargs="+", default=None, help="Filtra por valores de 'direction' (ej: 1 2 3 o long tra).")
    ap.add_argument("--brains", nargs="+", default=None, help="Brains a incluir (ej: BRAIN LUDG MBBL).")
    ap.add_argument("--out_root", default="analysis/ogse_experiments/contrast", help="directory root")
    ap.add_argument("--exp", default=None, help="override de sheet (solo naming)")
    args = ap.parse_args()

    directions = [str(x) for x in (args.direction or [])]
    brains = args.brains
    if brains is not None and len(brains) == 1 and str(brains[0]).upper() == "ALL":
        brains = None

    df_ref = pd.read_parquet(Path(args.ref_parquet))
    df_cmp = pd.read_parquet(Path(args.cmp_parquet))

    _validate_input(df_ref, "ref")
    _validate_input(df_cmp, "cmp")

    df_ref = _normalize_key_dtypes(df_ref, "ref")
    df_cmp = _normalize_key_dtypes(df_cmp, "cmp")

    if directions:
        df_ref = df_ref[df_ref["direction"].isin(directions)]
        df_cmp = df_cmp[df_cmp["direction"].isin(directions)]

    analysis_id, analysis_short = build_analysis_id(df_ref, df_cmp, directions, args.exp)
    sheet = canonical_sheet_name(args.exp or _one(df_ref, "sheet", _one(df_cmp, "sheet", None)))
    brain = infer_brain_group(sheet, source_name=analysis_id)

    if brains is not None and str(brain) not in {str(x) for x in brains}:
        print(f"Skipped: {analysis_id} (brain={brain})")
        return

    # Core contrast (devuelve value/value_norm, y value_1/value_2, etc.)
    res = make_contrast(
        df_ref,
        df_cmp,
        axes=tuple(directions) if directions else None,
        y_col="value",
        y_norm_col="signal_norm",  # en output se renombra a value_norm_1/_2
        key_cols=KEY_COLS,
    )
    out = res.df.copy()

    _validate_input(out, "contrast_out")
    out = _normalize_key_dtypes(out, "contrast_out")

    # Arrastrar TODAS las columnas extra desde ref y cmp
    out = _merge_side_columns(out, df_ref, side=1)
    out = _merge_side_columns(out, df_cmp, side=2)

    # Limpieza + generic naming + de-dup aliases
    out = _drop_legacy_cols(out)
    out = _rename_to_generic(out)
    out = _dedup_aliases(out)

    out["analysis_id"] = str(analysis_id)
    out["sheet"] = sheet
    out["brain"] = str(brain)

    # Orden final de columnas
    out = _order_columns(out)

    tables_dir = Path(args.out_root) / "tables" / analysis_short
    tables_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = tables_dir / f"{analysis_id}.long.parquet"
    out.to_parquet(out_parquet, index=False)
    out.to_excel(out_parquet.with_suffix(".xlsx"), index=False)

    print("Saved:", out_parquet)


if __name__ == "__main__":
    main()
