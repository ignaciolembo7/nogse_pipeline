from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PRINTABLE_RE = re.compile(rb"[\x09\x20-\x7E]{4,}")
KEY_VALUE_RE = re.compile(r"^(?P<key>[A-Za-z0-9_.\[\]<>/-]+)\s*=\s*(?P<value>.*)$")
FLOAT_LINE_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
G_TOKEN_RE = re.compile(r"(?:^|_)G(?P<g>\d+(?:p\d+)?)(?:_|$)", re.IGNORECASE)
NOGSE_TOKEN_RE = re.compile(
    r"(?P<group>\d{3})_NOGSE_(?P<seq_type>CPMG|HAHN)_N(?P<N>\d+(?:p\d+)?)_TN(?P<TN>\d+(?:p\d+)?)_G(?P<G>\d+(?:p\d+)?)",
    re.IGNORECASE,
)
NIFTI_SERIES_RE = re.compile(r"_(?P<series>\d+)[A-Za-z]*(?:\.nii(?:\.gz)?|$)")


@dataclass(frozen=True)
class DicomMeta:
    path: Path
    series_number: int | None
    image_number: int | None
    protocol_name: str
    sequence_file_name: str
    wip_g_mtm: float | None
    wip_g_source: str
    protocol_g_mtm: float | None
    stim_max_ges_norm_online: float | None
    stim_effective_g_mtm: float | None
    phase_gradient_amplitude: float | None
    readout_gradient_amplitude: float | None
    selection_gradient_amplitude: float | None
    asconv: dict[str, str]
    strings: list[str]


def parse_float_token(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().strip('"')
    text = text.replace("p", ".")
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def clean_text(value: str | None) -> str:
    if value is None:
        return ""
    out = str(value).strip()
    while len(out) >= 2 and out[0] == '"' and out[-1] == '"':
        out = out[1:-1].strip()
    out = out.replace('""', '"')
    return out


def extract_printable_strings(path: Path) -> list[str]:
    data = path.read_bytes()
    out: list[str] = []
    for match in PRINTABLE_RE.finditer(data):
        text = match.group(0).decode("latin1", errors="replace").strip()
        if text:
            out.append(text)
    return out


def parse_asconv(strings: list[str]) -> dict[str, str]:
    in_block = False
    out: dict[str, str] = {}
    for text in strings:
        if "### ASCCONV BEGIN" in text:
            in_block = True
            continue
        if "### ASCCONV END" in text:
            break
        if not in_block:
            continue
        match = KEY_VALUE_RE.match(text.strip())
        if match:
            out[match.group("key")] = clean_text(match.group("value"))
    return out


def first_numeric_after(strings: list[str], label: str, *, max_lookahead: int = 10) -> float | None:
    for idx, text in enumerate(strings):
        if label not in text:
            continue
        for candidate in strings[idx + 1 : idx + 1 + max_lookahead]:
            stripped = candidate.strip()
            if FLOAT_LINE_RE.match(stripped):
                return parse_float_token(stripped)
    return None


def parse_series_image_from_name(path: Path) -> tuple[int | None, int | None]:
    parts = path.name.split(".")
    if len(parts) < 5:
        return None, None
    series = int(parts[3]) if parts[3].isdigit() else None
    image = int(parts[4]) if parts[4].isdigit() else None
    return series, image


def parse_g_from_protocol(protocol_name: str) -> float | None:
    match = G_TOKEN_RE.search(protocol_name)
    if not match:
        return None
    return parse_float_token(match.group("g"))


def parse_protocol_tokens(protocol_name: str) -> dict[str, object]:
    match = NOGSE_TOKEN_RE.search(protocol_name)
    if not match:
        return {}
    return {
        "group": parse_float_token(match.group("group")),
        "type": match.group("seq_type").upper(),
        "N": parse_float_token(match.group("N")),
        "TN": parse_float_token(match.group("TN")),
        "G_from_protocol_name": parse_float_token(match.group("G")),
    }


def strip_nii_ext(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def parse_nifti_series_number(path: Path) -> int | None:
    match = NIFTI_SERIES_RE.search(path.name)
    if not match:
        return None
    try:
        return int(match.group("series"))
    except ValueError:
        return None


def parse_protocol_from_nifti_name(path: Path) -> str:
    name = strip_nii_ext(path)
    match = NOGSE_TOKEN_RE.search(name)
    return match.group(0) if match else name


def extract_dicom_meta(path: Path, *, scanner_grad_max_mtm: float) -> DicomMeta:
    strings = extract_printable_strings(path)
    asconv = parse_asconv(strings)
    series_number, image_number = parse_series_image_from_name(path)

    protocol_name = clean_text(asconv.get("tProtocolName"))
    if not protocol_name:
        protocol_name = next((s for s in strings if "NOGSE" in s and "_G" in s), "")

    sequence_file_name = clean_text(asconv.get("tSequenceFileName"))

    wip_g_raw = parse_float_token(asconv.get("sWipMemBlock.alFree[7]"))
    protocol_g = parse_g_from_protocol(protocol_name)
    if wip_g_raw is None and protocol_g == 0:
        wip_g = 0.0
        wip_source = "sWipMemBlock.alFree[7] missing; protocol G00 assumed zero"
    else:
        wip_g = wip_g_raw
        wip_source = "sWipMemBlock.alFree[7]" if wip_g_raw is not None else ""

    stim_norm = first_numeric_after(strings, "Stim_max_ges_norm_online")
    stim_effective = None
    if stim_norm is not None:
        stim_effective = float(stim_norm) * float(scanner_grad_max_mtm)

    return DicomMeta(
        path=path,
        series_number=series_number,
        image_number=image_number,
        protocol_name=protocol_name,
        sequence_file_name=sequence_file_name,
        wip_g_mtm=wip_g,
        wip_g_source=wip_source,
        protocol_g_mtm=protocol_g,
        stim_max_ges_norm_online=stim_norm,
        stim_effective_g_mtm=stim_effective,
        phase_gradient_amplitude=first_numeric_after(strings, "PhaseGradientAmplitude"),
        readout_gradient_amplitude=first_numeric_after(strings, "ReadoutGradientAmplitude"),
        selection_gradient_amplitude=first_numeric_after(strings, "SelectionGradientAmplitude"),
        asconv=asconv,
        strings=strings,
    )


def unique_finite(values: Iterable[float | None]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        x = float(value)
        if not math.isfinite(x):
            continue
        if not any(math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-9) for y in out):
            out.append(x)
    return sorted(out)


def unique_text(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return sorted(out)


def one_or_blank(values: list[float]) -> float | str:
    if len(values) == 1:
        return values[0]
    return ""


def build_dicom_rows(metas: list[DicomMeta], scanner_grad_max_mtm: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for meta in metas:
        rows.append(
            {
                "dicom_file": str(meta.path),
                "series": meta.series_number,
                "image": meta.image_number,
                "protocol_name": meta.protocol_name,
                "sequence_file_name": meta.sequence_file_name,
                "g_wip_mtm": meta.wip_g_mtm,
                "g_wip_source": meta.wip_g_source,
                "g_protocol_name_mtm": meta.protocol_g_mtm,
                "scanner_grad_max_mtm": scanner_grad_max_mtm,
                "g_wip_fraction_of_scanner_max": None
                if meta.wip_g_mtm is None
                else meta.wip_g_mtm / scanner_grad_max_mtm,
                "stim_max_ges_norm_online": meta.stim_max_ges_norm_online,
                "stim_effective_g_mtm": meta.stim_effective_g_mtm,
                "phase_gradient_amplitude": meta.phase_gradient_amplitude,
                "readout_gradient_amplitude": meta.readout_gradient_amplitude,
                "selection_gradient_amplitude": meta.selection_gradient_amplitude,
            }
        )
    return rows


def consistency_notes(value_sets: dict[str, list[float]]) -> str:
    notes = []
    for name, values in value_sets.items():
        if len(values) > 1:
            notes.append(f"{name} has multiple values: {values}")
    return "; ".join(notes)


def summarize_sequence_group(
    *,
    series_number: int | None,
    protocol_name: str,
    group: list[DicomMeta],
    scanner_grad_max_mtm: float,
    nifti_file: Path | None = None,
) -> dict[str, object]:
    tokens = parse_protocol_tokens(protocol_name)
    wip_values = unique_finite(m.wip_g_mtm for m in group)
    protocol_values = unique_finite(m.protocol_g_mtm for m in group)
    stim_values = unique_finite(m.stim_max_ges_norm_online for m in group)
    stim_effective_values = unique_finite(m.stim_effective_g_mtm for m in group)
    phase_values = unique_finite(m.phase_gradient_amplitude for m in group)
    readout_values = unique_finite(m.readout_gradient_amplitude for m in group)
    selection_values = unique_finite(m.selection_gradient_amplitude for m in group)
    source_values = unique_text(m.wip_g_source for m in group)

    g_wip = one_or_blank(wip_values)
    g_protocol = one_or_blank(protocol_values)
    stim_norm = one_or_blank(stim_values)
    stim_effective = one_or_blank(stim_effective_values)

    return {
        "nifti_file": "" if nifti_file is None else str(nifti_file),
        "nifti_base": "" if nifti_file is None else strip_nii_ext(nifti_file),
        "Protocol": protocol_name,
        "sequence": series_number,
        "group": tokens.get("group", ""),
        "G": g_wip,
        "G_source": "; ".join(source_values),
        "G_from_protocol_name": g_protocol,
        "G_fraction_of_scanner_max": ""
        if not isinstance(g_wip, float)
        else g_wip / scanner_grad_max_mtm,
        "G_stim_norm_online": stim_norm,
        "G_stim_effective_mTm": stim_effective,
        "G_stim_source": "Stim_max_ges_norm_online * scanner_grad_max_mtm"
        if isinstance(stim_effective, float)
        else "",
        "scanner_grad_max_mTm": scanner_grad_max_mtm,
        "TN": tokens.get("TN", ""),
        "N": tokens.get("N", ""),
        "type": tokens.get("type", ""),
        "dicom_count": len(group),
        "first_image": min((m.image_number for m in group if m.image_number is not None), default=""),
        "last_image": max((m.image_number for m in group if m.image_number is not None), default=""),
        "PhaseGradientAmplitude": one_or_blank(phase_values),
        "ReadoutGradientAmplitude": one_or_blank(readout_values),
        "SelectionGradientAmplitude": one_or_blank(selection_values),
        "consistency_notes": consistency_notes(
            {
                "G": wip_values,
                "G_from_protocol_name": protocol_values,
                "G_stim_norm_online": stim_values,
            }
        ),
    }


def build_sequence_rows(metas: list[DicomMeta], scanner_grad_max_mtm: float) -> list[dict[str, object]]:
    grouped: dict[tuple[int | None, str], list[DicomMeta]] = defaultdict(list)
    for meta in metas:
        grouped[(meta.series_number, meta.protocol_name)].append(meta)

    rows: list[dict[str, object]] = []
    for (series_number, protocol_name), group in sorted(grouped.items(), key=lambda item: (item[0][0] or -1, item[0][1])):
        rows.append(
            summarize_sequence_group(
                series_number=series_number,
                protocol_name=protocol_name,
                group=group,
                scanner_grad_max_mtm=scanner_grad_max_mtm,
            )
        )
    return rows


def build_nifti_rows(
    metas: list[DicomMeta],
    nifti_files: list[Path],
    scanner_grad_max_mtm: float,
) -> list[dict[str, object]]:
    by_series: dict[int | None, list[DicomMeta]] = defaultdict(list)
    for meta in metas:
        by_series[meta.series_number].append(meta)

    rows: list[dict[str, object]] = []
    for nifti_file in sorted(nifti_files):
        series_number = parse_nifti_series_number(nifti_file)
        group = by_series.get(series_number, [])
        if group:
            protocol_names = unique_text(m.protocol_name for m in group)
            protocol_name = protocol_names[0] if len(protocol_names) == 1 else parse_protocol_from_nifti_name(nifti_file)
            row = summarize_sequence_group(
                series_number=series_number,
                protocol_name=protocol_name,
                group=group,
                scanner_grad_max_mtm=scanner_grad_max_mtm,
                nifti_file=nifti_file,
            )
            if len(protocol_names) > 1:
                note = row.get("consistency_notes", "")
                row["consistency_notes"] = "; ".join(
                    x for x in [str(note), f"multiple DICOM protocol names for series: {protocol_names}"] if x
                )
        else:
            row = summarize_sequence_group(
                series_number=series_number,
                protocol_name=parse_protocol_from_nifti_name(nifti_file),
                group=[],
                scanner_grad_max_mtm=scanner_grad_max_mtm,
                nifti_file=nifti_file,
            )
            row["consistency_notes"] = "no matching DICOM metadata for this NIfTI series"
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(path: Path, rows: list[dict[str, object]]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Writing Parquet output requires pandas/pyarrow in the active environment.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def write_table_pair(csv_path: Path, rows: list[dict[str, object]], *, write_parquet_output: bool) -> Path | None:
    write_csv(csv_path, rows)
    parquet_path = csv_path.with_suffix(".parquet")
    if csv_path.name.endswith(".long.csv"):
        parquet_path = csv_path.with_name(csv_path.name[: -len(".long.csv")] + ".long.parquet")
    if write_parquet_output:
        write_parquet(parquet_path, rows)
        return parquet_path
    return None


def build_key_value_rows(metas: list[DicomMeta]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for meta in metas:
        for key, value in sorted(meta.asconv.items()):
            rows.append(
                {
                    "dicom_file": str(meta.path),
                    "series": meta.series_number,
                    "image": meta.image_number,
                    "source": "ASCCONV",
                    "key": key,
                    "value": value,
                }
            )
    return rows


def build_string_rows(metas: list[DicomMeta]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for meta in metas:
        for idx, value in enumerate(meta.strings, start=1):
            rows.append(
                {
                    "dicom_file": str(meta.path),
                    "series": meta.series_number,
                    "image": meta.image_number,
                    "string_index": idx,
                    "value": value,
                }
            )
    return rows


def write_excel(path: Path, sheets: dict[str, list[dict[str, object]]]) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("Writing Excel output requires pandas/openpyxl in the active environment.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, rows in sheets.items():
            pd.DataFrame(rows).to_excel(writer, sheet_name=sheet, index=False)


def discover_files(root: Path, patterns: list[str], recursive: bool) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        files.extend(path for path in iterator if path.is_file())
    return sorted(set(files))


@dataclass(frozen=True)
class ExtractionOutputs:
    sequence_csv: Path
    sequence_parquet: Path | None
    nifti_csv: Path | None
    nifti_parquet: Path | None
    dicom_csv: Path
    dicom_parquet: Path | None
    key_values_csv: Path
    key_values_parquet: Path | None
    strings_csv: Path | None
    strings_parquet: Path | None
    xlsx: Path | None


def extract_metadata_tables(
    *,
    dicom_root: Path,
    out_root: Path,
    dicom_patterns: list[str],
    dicom_recursive: bool,
    scanner_grad_max_mtm: float,
    nifti_root: Path | None = None,
    nifti_patterns: list[str] | None = None,
    nifti_recursive: bool = False,
    write_strings: bool = False,
    out_xlsx: Path | None = None,
    write_parquet_output: bool = True,
) -> ExtractionOutputs:
    if not dicom_root.is_dir():
        raise SystemExit(f"Input DICOM root does not exist: {dicom_root}")
    if scanner_grad_max_mtm <= 0:
        raise SystemExit("--scanner-grad-max-mtm must be positive.")

    dicom_files = discover_files(dicom_root, dicom_patterns, dicom_recursive)
    if not dicom_files:
        raise SystemExit(f"No DICOM files matched {dicom_patterns!r} in {dicom_root}")

    metas = [extract_dicom_meta(path, scanner_grad_max_mtm=scanner_grad_max_mtm) for path in dicom_files]
    dicom_rows = build_dicom_rows(metas, scanner_grad_max_mtm)
    sequence_rows = build_sequence_rows(metas, scanner_grad_max_mtm)
    nifti_rows: list[dict[str, object]] = []

    if nifti_root is not None:
        if not nifti_root.is_dir():
            raise SystemExit(f"Input NIfTI root does not exist: {nifti_root}")
        patterns = nifti_patterns or ["*.nii.gz"]
        nifti_files = discover_files(nifti_root, patterns, nifti_recursive)
        if not nifti_files:
            raise SystemExit(f"No NIfTI files matched {patterns!r} in {nifti_root}")
        nifti_rows = build_nifti_rows(metas, nifti_files, scanner_grad_max_mtm)

    sequence_csv = out_root / "sequence_parameters_from_dicom.csv"
    nifti_csv = out_root / "sequence_parameters_by_nifti_from_dicom.csv" if nifti_rows else None
    dicom_csv = out_root / "dicom_metadata_summary.csv"
    key_values_csv = out_root / "dicom_asconv_key_values.long.csv"

    sequence_parquet = write_table_pair(sequence_csv, sequence_rows, write_parquet_output=write_parquet_output)
    nifti_parquet = (
        write_table_pair(nifti_csv, nifti_rows, write_parquet_output=write_parquet_output)
        if nifti_csv is not None
        else None
    )
    dicom_parquet = write_table_pair(dicom_csv, dicom_rows, write_parquet_output=write_parquet_output)
    key_values_parquet = write_table_pair(
        key_values_csv,
        build_key_value_rows(metas),
        write_parquet_output=write_parquet_output,
    )

    strings_csv = None
    strings_parquet = None
    if write_strings:
        strings_csv = out_root / "dicom_printable_strings.long.csv"
        strings_parquet = write_table_pair(
            strings_csv,
            build_string_rows(metas),
            write_parquet_output=write_parquet_output,
        )

    if out_xlsx is not None:
        write_excel(
            out_xlsx,
            {
                "sequence_parameters": sequence_rows,
                "sequence_by_nifti": nifti_rows,
                "dicom_metadata": dicom_rows,
            },
        )

    return ExtractionOutputs(
        sequence_csv=sequence_csv,
        sequence_parquet=sequence_parquet,
        nifti_csv=nifti_csv,
        nifti_parquet=nifti_parquet,
        dicom_csv=dicom_csv,
        dicom_parquet=dicom_parquet,
        key_values_csv=key_values_csv,
        key_values_parquet=key_values_parquet,
        strings_csv=strings_csv,
        strings_parquet=strings_parquet,
        xlsx=out_xlsx,
    )
