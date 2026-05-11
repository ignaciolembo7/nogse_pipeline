from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


def parse_numeric_text(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().strip('"')
    if not text:
        return None
    text = text.replace("p", ".")
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def read_key_value_chunks(path: Path, chunksize: int, *, columns: list[str]):
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Reading DICOM parameter tables requires pandas in the active environment.") from exc

    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=chunksize, columns=columns):
            yield batch.to_pandas()
    else:
        yield from pd.read_csv(path, usecols=columns, chunksize=chunksize)


def match_dicom_rows(chunk, query: str):
    query_path = Path(query)
    query_name = query_path.name
    query_stem = query_path.stem
    files = chunk["dicom_file"].astype(str)
    basenames = files.str.rsplit("/", n=1).str[-1]
    stems = basenames.str.replace(r"\.[^.]+$", "", regex=True)
    return chunk.loc[
        (files == query)
        | (basenames == query_name)
        | (stems == query_stem)
        | files.str.contains(query, regex=False)
        | basenames.str.contains(query_name, regex=False)
    ]


def default_output_stem(query: str) -> str:
    return Path(query).name.replace("/", "_")


@dataclass(frozen=True)
class SingleDicomExportOutputs:
    parquet: Path
    xlsx: Path
    csv: Path | None
    matched_files: int
    rows: int


def export_one_dicom_parameters(
    *,
    key_values: Path,
    dicom_file: str,
    out_parquet: Path | None = None,
    out_xlsx: Path | None = None,
    out_csv: Path | None = None,
    chunksize: int = 250_000,
    progress_every: int = 0,
) -> SingleDicomExportOutputs:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Exporting DICOM parameter tables requires pandas/openpyxl/pyarrow.") from exc

    if not key_values.is_file():
        raise SystemExit(f"Key-value table does not exist: {key_values}")

    matches = []
    columns = ["dicom_file", "series", "image", "source", "key", "value"]
    for idx, chunk in enumerate(read_key_value_chunks(key_values, chunksize, columns=columns), start=1):
        selected = match_dicom_rows(chunk, dicom_file)
        if not selected.empty:
            matches.append(selected.copy())
        if progress_every > 0 and idx % int(progress_every) == 0:
            print(f"[INFO] Scanned chunk {idx}")

    if not matches:
        raise SystemExit(f"No DICOM rows matched: {dicom_file}")

    out = pd.concat(matches, ignore_index=True)
    out["value_numeric"] = out["value"].map(parse_numeric_text)
    out = out.sort_values(["dicom_file", "source", "key"], kind="stable")

    safe_name = default_output_stem(dicom_file)
    parquet_path = out_parquet or key_values.with_name(f"{safe_name}.dicom_parameters.long.parquet")
    xlsx_path = out_xlsx or key_values.with_name(f"{safe_name}.dicom_parameters.long.xlsx")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    summary = (
        out[["dicom_file", "series", "image"]]
        .drop_duplicates()
        .sort_values(["dicom_file", "series", "image"], kind="stable")
    )
    numeric = out.loc[out["value_numeric"].notna()].copy()

    if out_csv is not None:
        out.to_csv(out_csv, index=False)
    out.to_parquet(parquet_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        out.to_excel(writer, sheet_name="parameters_long", index=False)
        numeric.to_excel(writer, sheet_name="numeric_parameters", index=False)
        summary.to_excel(writer, sheet_name="matched_dicoms", index=False)

    return SingleDicomExportOutputs(
        parquet=parquet_path,
        xlsx=xlsx_path,
        csv=out_csv,
        matched_files=int(summary["dicom_file"].nunique()),
        rows=int(len(out)),
    )
