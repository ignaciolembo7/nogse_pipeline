from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class ContrastResult:
    df: pd.DataFrame


def make_contrast(
    df_ref: pd.DataFrame,
    df_cmp: pd.DataFrame,
    *,
    axes: tuple[str, ...] | None = ("long", "tra"),  # en tu pipeline esto son "directions"
    y_col: str = "value",
    y_norm_col: str = "signal_norm",  # lo transformamos a "value_norm" en el script CLI
    key_cols: tuple[str, ...] = ("stat", "roi", "direction", "b_step"),
) -> ContrastResult:
    """
    Devuelve una tabla long con sufijos _1/_2 y dos columnas GENÉRICAS:

      value      = y_ref - y_cmp
      value_norm = y_norm_ref - y_norm_cmp   (si existe y_norm en ambas)
                = (y_ref/S0_ref) - (y_cmp/S0_cmp)  (fallback)

    Reglas duras:
      - la columna de direction se llama SIEMPRE 'direction' (nunca 'axis')
      - no arrastra param_ / meta_ (eso se hace upstream si querés, pero tu nuevo estándar es limpio)
    """
    a = df_ref.copy()
    b = df_cmp.copy()

    # --- Regla: direction SIEMPRE, axis NUNCA
    if "axis" in a.columns or "axis" in b.columns:
        raise KeyError("Encontré columna 'axis'. En este pipeline SOLO se permite 'direction'.")
    if "direction" not in a.columns or "direction" not in b.columns:
        raise KeyError("Falta columna obligatoria 'direction' en una de las tablas.")

    # --- Key cols requeridas
    for name, df in [("df_ref", a), ("df_cmp", b)]:
        missing = [c for c in key_cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name}: faltan columnas clave {missing}. Necesito {key_cols}.")

    # --- Validar columnas de señal
    if y_col not in a.columns or y_col not in b.columns:
        raise KeyError(f"Falta y_col='{y_col}' en ref/cmp. (No hay fallback a 'signal'.)")

    # --- Filtrar direcciones si se pidió
    if axes is not None:
        axes = tuple(str(x) for x in axes)
        a["direction"] = a["direction"].astype(str)
        b["direction"] = b["direction"].astype(str)
        a = a[a["direction"].isin(axes)]
        b = b[b["direction"].isin(axes)]

    # --- Columnas mínimas para el merge base
    keep_cols = set(key_cols)
    keep_cols.add(y_col)

    have_norm = (y_norm_col in a.columns) and (y_norm_col in b.columns)
    if have_norm:
        keep_cols.add(y_norm_col)

    have_S0 = ("S0" in a.columns) and ("S0" in b.columns)
    if have_S0:
        keep_cols.add("S0")

    a = a[[c for c in keep_cols if c in a.columns]].copy()
    b = b[[c for c in keep_cols if c in b.columns]].copy()

    # --- Merge
    m = a.merge(b, on=list(key_cols), suffixes=("_1", "_2"), how="inner")

    # --- value (GENÉRICO)
    m["value"] = m[f"{y_col}_1"] - m[f"{y_col}_2"]

    # --- value_norm (GENÉRICO)
    if have_norm and f"{y_norm_col}_1" in m.columns and f"{y_norm_col}_2" in m.columns:
        m["value_norm"] = m[f"{y_norm_col}_1"] - m[f"{y_norm_col}_2"]
    else:
        if "S0_1" not in m.columns or "S0_2" not in m.columns:
            raise KeyError(
                "No existe y_norm_col en ambas tablas y tampoco existen S0_1/S0_2 para fallback."
            )
        m["value_norm"] = (m[f"{y_col}_1"] / m["S0_1"]) - (m[f"{y_col}_2"] / m["S0_2"])

    # --- Orden estable de filas
    sort_cols = [c for c in ["stat", "roi", "direction", "b_step"] if c in m.columns]
    if sort_cols:
        m = m.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    return ContrastResult(df=m)