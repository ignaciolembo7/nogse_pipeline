from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import pandas as pd

from data_processing.io import write_xlsx_sheets


@dataclass(frozen=True)
class MetaVectors:
    d: list[float | str] | None = None
    Delta: list[float | str] | None = None
    Hz: list[float | str] | None = None
    max_dur: list[float | str] | None = None


def _parse_vector(value: str | None) -> list[float | str] | None:
    if value is None or str(value).strip() == "":
        return None
    tokens = [tok for tok in re.split(r"[\s,]+", str(value).strip()) if tok]
    out: list[float | str] = []
    for token in tokens:
        try:
            out.append(float(token))
        except ValueError:
            out.append(token)
    return out


def _format_token(value: object) -> str:
    if value is None:
        return "NA"
    try:
        num = float(value)
    except (TypeError, ValueError):
        text = str(value)
    else:
        if pd.isna(num):
            return "NA"
        if abs(num - round(num)) < 1e-9:
            text = str(int(round(num)))
        else:
            text = f"{num:.6g}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text.replace(".", "p")).strip("_") or "NA"


class _FilenameValue:
    def __init__(self, value: object) -> None:
        self.value = value

    def __format__(self, spec: str) -> str:
        if spec:
            return format(self.value, spec)
        return _format_token(self.value)


def format_name(template: str, **values: object) -> str:
    formatted = {key: _FilenameValue(value) for key, value in values.items()}
    return template.format(**formatted)


def parse_meta_vectors(
    *,
    d: str | None = None,
    Delta: str | None = None,
    Hz: str | None = None,
    max_dur: str | None = None,
) -> MetaVectors:
    return MetaVectors(
        d=_parse_vector(d),
        Delta=_parse_vector(Delta),
        Hz=_parse_vector(Hz),
        max_dur=_parse_vector(max_dur),
    )


def validate_meta_lengths(n_experiments: int, meta: MetaVectors) -> None:
    for name in ("d", "Delta", "Hz", "max_dur"):
        values = getattr(meta, name)
        if values is not None and len(values) != n_experiments:
            raise ValueError(
                f"Metadata vector {name!r} has {len(values)} values, expected {n_experiments}."
            )


def read_results_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_bvals_txt(path: str | Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append([float(tok) for tok in re.split(r"[\s,;]+", stripped) if tok])
    return rows


def split_experiments(df: pd.DataFrame, *, chunk_size: int) -> list[pd.DataFrame]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if len(df) % chunk_size != 0:
        raise ValueError(
            f"Results table has {len(df)} rows, which is not divisible by chunk_size={chunk_size}."
        )
    return [df.iloc[start : start + chunk_size].copy() for start in range(0, len(df), chunk_size)]


def _stat_columns(df: pd.DataFrame, stat: str) -> dict[str, str]:
    prefix = f"{stat}("
    suffix = ")"
    matches = {
        col[len(prefix) : -len(suffix)]: col
        for col in df.columns
        if str(col).startswith(prefix) and str(col).endswith(suffix)
    }
    if matches:
        return matches

    if stat in df.columns:
        return {stat: stat}

    raise ValueError(f"Could not find columns for stat {stat!r}. Available columns: {list(df.columns)}")


def make_tables_for_experiment(
    chunk: pd.DataFrame,
    *,
    bvals: Sequence[float],
    stats: Sequence[str],
) -> dict[str, pd.DataFrame]:
    if len(bvals) != len(chunk):
        raise ValueError(f"bvals has {len(bvals)} values, expected {len(chunk)} rows.")

    tables: dict[str, pd.DataFrame] = {}
    for stat in stats:
        roi_cols = _stat_columns(chunk, stat)
        table = pd.DataFrame({"bvalues": list(bvals)})
        for roi_name, col in roi_cols.items():
            table[roi_name] = pd.to_numeric(chunk[col], errors="coerce")
        tables[str(stat)] = table
    return tables


def write_experiment_xlsx(out_path: str | Path, tables: dict[str, pd.DataFrame]) -> None:
    write_xlsx_sheets(tables, out_path)
