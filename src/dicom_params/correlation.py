from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from dicom_params.single_file import parse_numeric_text, read_key_value_chunks


G_TOKEN_RE = re.compile(r"(?:^|_)G(?P<g>\d+(?:p\d+)?)(?:_|$)", re.IGNORECASE)


def parse_gradient_from_name(value: object) -> float | None:
    match = G_TOKEN_RE.search(str(value))
    if not match:
        return None
    return parse_numeric_text(match.group("g"))


def load_series_gradient_map(nifti_table: Path) -> tuple[dict[int, float], list[str]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("This script requires pandas in the active environment.") from exc

    if nifti_table.suffix == ".parquet":
        df = pd.read_parquet(nifti_table)
    else:
        df = pd.read_csv(nifti_table)
    required = {"sequence"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{nifti_table} is missing required columns: {sorted(missing)}")

    if "G" in df.columns:
        gradients = df["G"].map(parse_numeric_text)
    elif "nifti_base" in df.columns:
        gradients = df["nifti_base"].map(parse_gradient_from_name)
    elif "nifti_file" in df.columns:
        gradients = df["nifti_file"].map(parse_gradient_from_name)
    else:
        raise SystemExit(f"{nifti_table} needs column G, nifti_base, or nifti_file to identify gradients.")

    work = df.assign(_series=df["sequence"].map(parse_numeric_text), _gradient=gradients)
    work = work.dropna(subset=["_series", "_gradient"])
    work["_series"] = work["_series"].astype(int)

    series_to_gradient: dict[int, float] = {}
    notes: list[str] = []
    for series, group in work.groupby("_series"):
        values = sorted({float(value) for value in group["_gradient"]})
        if len(values) == 1:
            series_to_gradient[int(series)] = values[0]
        else:
            notes.append(f"Series {int(series)} has multiple gradients in NIfTI table: {values}")
    return series_to_gradient, notes


def make_empty_stats() -> dict[str, float]:
    return {
        "n": 0.0,
        "sum_x": 0.0,
        "sum_y": 0.0,
        "sum_x2": 0.0,
        "sum_y2": 0.0,
        "sum_xy": 0.0,
        "min_value": math.inf,
        "max_value": -math.inf,
        "min_gradient": math.inf,
        "max_gradient": -math.inf,
    }


def update_stats(stats: dict[str, dict[str, float]], chunk, series_to_gradient: dict[int, float], source: str) -> None:
    import pandas as pd

    if source:
        chunk = chunk.loc[chunk["source"].astype(str) == source]
    if chunk.empty:
        return

    chunk = chunk.copy()
    chunk["_series"] = pd.to_numeric(chunk["series"], errors="coerce")
    chunk["_gradient"] = chunk["_series"].map(lambda x: series_to_gradient.get(int(x)) if pd.notna(x) else None)
    chunk["_value_numeric"] = chunk["value"].map(parse_numeric_text)
    chunk = chunk.dropna(subset=["_gradient", "_value_numeric"])
    if chunk.empty:
        return

    chunk["_x2"] = chunk["_value_numeric"] * chunk["_value_numeric"]
    chunk["_y2"] = chunk["_gradient"] * chunk["_gradient"]
    chunk["_xy"] = chunk["_value_numeric"] * chunk["_gradient"]

    grouped = chunk.groupby("key", dropna=False).agg(
        n=("key", "size"),
        sum_x=("_value_numeric", "sum"),
        sum_y=("_gradient", "sum"),
        sum_x2=("_x2", "sum"),
        sum_y2=("_y2", "sum"),
        sum_xy=("_xy", "sum"),
        min_value=("_value_numeric", "min"),
        max_value=("_value_numeric", "max"),
        min_gradient=("_gradient", "min"),
        max_gradient=("_gradient", "max"),
    )

    for key, row in grouped.iterrows():
        entry = stats[str(key)]
        entry["n"] += float(row["n"])
        entry["sum_x"] += float(row["sum_x"])
        entry["sum_y"] += float(row["sum_y"])
        entry["sum_x2"] += float(row["sum_x2"])
        entry["sum_y2"] += float(row["sum_y2"])
        entry["sum_xy"] += float(row["sum_xy"])
        entry["min_value"] = min(entry["min_value"], float(row["min_value"]))
        entry["max_value"] = max(entry["max_value"], float(row["max_value"]))
        entry["min_gradient"] = min(entry["min_gradient"], float(row["min_gradient"]))
        entry["max_gradient"] = max(entry["max_gradient"], float(row["max_gradient"]))


def finalize_stats(stats: dict[str, dict[str, float]], min_observations: int):
    import pandas as pd

    rows: list[dict[str, object]] = []
    for key, entry in stats.items():
        n = int(entry["n"])
        if n < min_observations:
            continue
        numerator = n * entry["sum_xy"] - entry["sum_x"] * entry["sum_y"]
        denom_x = n * entry["sum_x2"] - entry["sum_x"] * entry["sum_x"]
        denom_y = n * entry["sum_y2"] - entry["sum_y"] * entry["sum_y"]
        correlation = math.nan
        if denom_x > 0 and denom_y > 0:
            correlation = numerator / math.sqrt(denom_x * denom_y)
        rows.append(
            {
                "key": key,
                "correlation": correlation,
                "abs_correlation": abs(correlation) if math.isfinite(correlation) else math.nan,
                "n_observations": n,
                "min_value": entry["min_value"],
                "max_value": entry["max_value"],
                "min_gradient": entry["min_gradient"],
                "max_gradient": entry["max_gradient"],
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class CorrelationOutputs:
    csv: Path
    xlsx: Path
    rows: int
    series_count: int


def correlate_asconv_with_gradient(
    *,
    key_values: Path,
    nifti_table: Path,
    out_csv: Path | None = None,
    out_xlsx: Path | None = None,
    source: str = "ASCCONV",
    chunksize: int = 250_000,
    min_observations: int = 3,
    sort_by: str = "abs_correlation",
    progress_every: int = 0,
) -> CorrelationOutputs:
    import pandas as pd

    if not key_values.is_file():
        raise SystemExit(f"Key-value table does not exist: {key_values}")
    if not nifti_table.is_file():
        raise SystemExit(f"NIfTI table does not exist: {nifti_table}")

    csv_path = out_csv or key_values.with_name("dicom_asconv_gradient_correlations.csv")
    xlsx_path = out_xlsx or key_values.with_name("dicom_asconv_gradient_correlations.xlsx")

    series_to_gradient, notes = load_series_gradient_map(nifti_table)
    if not series_to_gradient:
        raise SystemExit("No series-to-gradient mappings were found.")

    stats = defaultdict(make_empty_stats)
    columns = ["series", "source", "key", "value"]
    for idx, chunk in enumerate(read_key_value_chunks(key_values, chunksize, columns=columns), start=1):
        update_stats(stats, chunk, series_to_gradient, source)
        if progress_every > 0 and idx % int(progress_every) == 0:
            print(f"[INFO] Processed chunk {idx}")

    df = finalize_stats(stats, min_observations)
    if df.empty:
        raise SystemExit("No numeric DICOM parameters had enough observations for correlation.")

    sort_columns = [sort_by, "n_observations", "key"]
    df = df.sort_values(sort_columns, ascending=[False, False, True], na_position="last")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="correlations", index=False)
        pd.DataFrame(
            [{"series": series, "gradient": gradient} for series, gradient in sorted(series_to_gradient.items())]
        ).to_excel(writer, sheet_name="series_gradient_map", index=False)
        if notes:
            pd.DataFrame({"note": notes}).to_excel(writer, sheet_name="notes", index=False)

    return CorrelationOutputs(
        csv=csv_path,
        xlsx=xlsx_path,
        rows=int(len(df)),
        series_count=int(len(series_to_gradient)),
    )
